import { CARE_MESSAGES as WORDS } from "./care-messages.js";
import { mountAccessibility } from "./accessibility.js";
import { summarizeDoses } from "./care-summary.js";

const $ = (id) => document.getElementById(id);
const escape = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
let language = "en";
try {
  language = localStorage.getItem("behaviorsense.care-language") || "en";
} catch {}
if (!["en", "hi", "mr"].includes(language)) language = "en";
const t = (key) =>
  (WORDS[key] || [key, key, key])[["en", "hi", "mr"].indexOf(language)];
let session = null;
try {
  session = JSON.parse(
    sessionStorage.getItem("behaviorsense.care-session") || "null",
  );
} catch {}
const state = {
  profile: null,
  medications: [],
  doses: [],
  appointments: [],
  outbox: [],
  history: [],
  mode: "preview",
  medEdit: null,
  appointmentEdit: null,
  profileDirty: false,
  profileFormVersion: null,
};
const value = (id) => $(id).value.trim();
function announce(message, error = false) {
  $("care-status").textContent = message;
  $("care-status").classList.toggle("error", error);
  $("care-status").setAttribute("aria-live", error ? "assertive" : "polite");
}
function localParts(when = new Date()) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: state.profile?.timezone || "Asia/Kolkata",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(when);
  const p = Object.fromEntries(parts.map((x) => [x.type, x.value]));
  return {
    day: `${p.year}-${p.month}-${p.day}`,
    time: `${p.hour}:${p.minute}`,
  };
}
function when(iso) {
  return new Intl.DateTimeFormat(
    { en: "en-IN", hi: "hi-IN", mr: "mr-IN" }[language],
    {
      timeZone: state.profile?.timezone || "Asia/Kolkata",
      dateStyle: "medium",
      timeStyle: "short",
    },
  ).format(new Date(iso));
}
async function api(path, method = "GET", body) {
  const response = await fetch(`/api/care/${path}`, {
    method,
    headers: {
      ...(session ? { Authorization: `Bearer ${session.token}` } : {}),
      ...(body ? { "Content-Type": "application/json" } : {}),
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  if (response.status === 204) return null;
  const data = await response.json();
  if (!response.ok) {
    if (response.status === 401) {
      session = null;
      sessionStorage.removeItem("behaviorsense.care-session");
      $("care-auth").hidden = false;
      $("care-workspace").hidden = true;
    }
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => item.msg).join("; ")
      : data.detail;
    throw new Error(detail || t("error"));
  }
  return data;
}
function saveSession(data) {
  session = { token: data.token, caregiver: data.caregiver };
  sessionStorage.setItem("behaviorsense.care-session", JSON.stringify(session));
  state.profile = data.profile;
}
function translations() {
  document.documentElement.lang = language;
  $("care-language").value = language;
  for (const el of document.querySelectorAll("[data-care-text]"))
    el.textContent = t(el.dataset.careText);
  mountAccessibility(language);
}
function badge(status) {
  return `<span class="care-badge ${escape(status)}">${escape(t(status))}</span>`;
}
function render() {
  const p = state.profile;
  if (!p) return;
  $("care-auth").hidden = true;
  $("care-workspace").hidden = false;
  $("profile-heading").textContent = `${p.name} · ${t("carePlan")}`;
  $("profile-meta").textContent =
    `${t("profileId")}: ${p.id} · ${p.timezone} · ${session.caregiver}`;
  if (!state.profileDirty) {
    $("profile-name").value = p.name;
    $("profile-email").value = p.email;
    $("profile-allergies").value = p.allergies;
    $("profile-note").value = p.care_note;
    $("profile-zone").value = p.timezone;
    $("profile-reminders").checked = p.reminders_enabled;
    state.profileFormVersion = p.version;
  }
  $("care-day").textContent = `${localParts().day} · ${p.timezone}`;
  const counts = summarizeDoses(state.doses);
  $("daily-confirmations").innerHTML =
    `<p><strong>${counts.confirmed} / ${counts.scheduled}</strong> ${escape(t("confirmedTotal"))}</p><dl>${["taken", "skipped", "pending", "overdue"].map((key) => `<div><dt>${escape(t(key))}</dt><dd data-dose-count="${key}">${counts[key]}</dd></div>`).join("")}</dl><p class="care-help">${escape(t("summaryHelp"))}</p>`;
  $("dose-list").innerHTML = state.doses.length
    ? state.doses
        .map((d) => {
          const confirmed = ["taken", "skipped"].includes(d.status),
            early = new Date(d.due_at).getTime() > Date.now() + 900000;
          const noteId = `dose-note-${d.medication_id}-${d.time.replace(":", "")}`;
          const note =
            confirmed && d.note
              ? `<p>${escape(t("note"))}: ${escape(d.note)}</p>`
              : !confirmed && !early
                ? `<label for="${noteId}">${escape(t("confirmationNote"))}: ${escape(d.name)}, ${escape(d.time)}</label><textarea id="${noteId}" data-dose-note maxlength="600" rows="2"></textarea>`
                : "";
          return `<article class="care-item"><div><h3>${escape(d.name)}</h3><p>${escape(d.dose)} · ${escape(d.time)} (${escape(p.timezone)})</p><p>${escape(d.instructions)}</p>${badge(d.status)}${confirmed ? `<p>${escape(t("reportedBy"))}: ${escape(d.caregiver)} · ${escape(when(d.confirmed_at))}</p>` : ""}${note}${early ? `<p class="care-help">${escape(t("dueLater"))}</p>` : ""}</div>${!confirmed ? `<div class="care-actions"><button type="button" data-action="taken" data-id="${d.medication_id}" data-time="${d.time}" data-day="${d.day}" ${early ? "disabled" : ""} aria-label="${escape(t("confirmTaken"))}: ${escape(d.name)}, ${d.time}">${escape(t("confirmTaken"))}</button><button type="button" class="secondary" data-action="skipped" data-id="${d.medication_id}" data-time="${d.time}" data-day="${d.day}" ${early ? "disabled" : ""} aria-label="${escape(t("confirmSkipped"))}: ${escape(d.name)}, ${d.time}">${escape(t("confirmSkipped"))}</button></div>` : ""}</article>`;
        })
        .join("")
    : `<p class="empty">${escape(t("noDoses"))}</p>`;
  $("medication-list").innerHTML = state.medications.length
    ? state.medications
        .map(
          (m) =>
            `<article class="care-item"><div><h3>${escape(m.name)}</h3><p>${escape(m.dose)} · ${escape(m.times.join(", "))}</p><p>${escape(m.instructions)}</p>${!m.active ? badge("inactive") : ""}</div><div class="care-actions"><button type="button" class="secondary" data-action="edit-medication" data-id="${m.id}" aria-label="${escape(t("edit"))}: ${escape(m.name)}">${escape(t("edit"))}</button>${m.active ? `<button type="button" class="secondary" data-action="stop-medication" data-id="${m.id}" aria-label="${escape(t("stop"))}: ${escape(m.name)}">${escape(t("stop"))}</button>` : ""}</div></article>`,
        )
        .join("")
    : `<p class="empty">${escape(t("noMedications"))}</p>`;
  $("appointment-list").innerHTML = state.appointments.length
    ? state.appointments
        .map(
          (a) =>
            `<article class="care-item"><div><h3>${escape(a.title)}</h3><p>${escape(a.doctor)} · ${escape(when(a.due_at))}</p><p>${escape(a.location)}</p><p>${escape(a.notes)}</p>${badge(a.status)}</div><div class="care-actions"><button type="button" class="secondary" data-action="edit-appointment" data-id="${a.id}" aria-label="${escape(t("edit"))}: ${escape(a.title)}">${escape(t("edit"))}</button>${a.status === "scheduled" ? `<button type="button" data-action="attended" data-id="${a.id}">${escape(t("markAttended"))}</button><button type="button" class="secondary" data-action="cancel-appointment" data-id="${a.id}">${escape(t("cancelAppointment"))}</button>` : ""}</div></article>`,
        )
        .join("")
    : `<p class="empty">${escape(t("noAppointments"))}</p>`;
  $("reminder-mode").textContent = t(
    state.mode === "smtp" ? "smtpHelp" : "previewHelp",
  );
  $("reminder-list").innerHTML = state.outbox.length
    ? state.outbox
        .map(
          (m) =>
            `<article class="care-item"><div><h3>${escape(m.subject)}</h3><p>${escape(m.recipient)} · ${escape(when(m.created_at))}</p>${badge(m.status)}<details><summary>${escape(t("preview"))}</summary><p>${escape(m.body).replace(/\n/g, "<br>")}</p>${m.error ? `<p>${escape(m.error)}</p>` : ""}</details></div></article>`,
        )
        .join("")
    : `<p class="empty">${escape(t("noReminders"))}</p>`;
  $("care-history").innerHTML = state.history.length
    ? `<div class="care-table-wrap" tabindex="0" role="region" aria-label="${escape(t("history"))}"><table><caption>${escape(t("history"))}</caption><thead><tr>${["day", "medication", "dose", "status", "reportedBy", "recordedAt", "note"].map((key) => `<th scope="col">${escape(t(key))}</th>`).join("")}</tr></thead><tbody>${state.history.map((r) => `<tr><td>${escape(r.day)} ${escape(r.time)}</td><td>${escape(r.medication_name)}</td><td>${escape(r.dose)}</td><td>${escape(t(r.status))}</td><td>${escape(r.caregiver)}</td><td>${escape(when(r.confirmed_at))}</td><td>${escape(r.note)}</td></tr>`).join("")}</tbody></table></div>`
    : `<p class="empty">${escape(t("noHistory"))}</p>`;
  if (!state.medEdit) $("medication-start").value = localParts().day;
  translations();
}
async function load() {
  const [profile, medications, doses, appointments, reminders, history] =
    await Promise.all([
      api("profile"),
      api("medications"),
      api("doses"),
      api("appointments"),
      api("reminders"),
      api("history"),
    ]);
  Object.assign(state, {
    profile,
    medications,
    doses,
    appointments,
    outbox: reminders.messages,
    mode: reminders.mode,
    history,
  });
  render();
}
function errorFor(id, error) {
  const el = $(id);
  el.textContent = `${t("error")} ${error.message}`;
  el.hidden = false;
  el.focus();
}
function bindForm(id, errorId, handler) {
  $(id).addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = $(id).querySelector('button[type="submit"]');
    button.disabled = true;
    $(errorId).hidden = true;
    try {
      await handler();
      announce(t("saved"));
    } catch (error) {
      errorFor(errorId, error);
    } finally {
      button.disabled = false;
    }
  });
}
function medBody() {
  return {
    name: value("medication-name"),
    dose: value("medication-dose"),
    instructions: value("medication-instructions"),
    times: value("medication-times")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean),
    start_day: value("medication-start"),
    active: $("medication-active").checked,
    ...(state.medEdit ? { version: state.medEdit.version } : {}),
  };
}
function appointmentBody() {
  return {
    title: value("appointment-title"),
    doctor: value("appointment-doctor"),
    local_at: value("appointment-time"),
    location: value("appointment-location"),
    notes: value("appointment-notes"),
    status: value("appointment-status"),
    ...(state.appointmentEdit
      ? { version: state.appointmentEdit.version }
      : {}),
  };
}
function editMedication(m) {
  state.medEdit = m;
  $("medication-name").value = m.name;
  $("medication-dose").value = m.dose;
  $("medication-times").value = m.times.join(", ");
  $("medication-start").value = m.start_day;
  $("medication-instructions").value = m.instructions;
  $("medication-active").checked = m.active;
  $("medication-editor").open = true;
  $("medication-name").focus();
}
function editAppointment(a) {
  state.appointmentEdit = a;
  $("appointment-title").value = a.title;
  $("appointment-doctor").value = a.doctor;
  $("appointment-time").value = a.local_at;
  $("appointment-location").value = a.location;
  $("appointment-notes").value = a.notes;
  $("appointment-status").value = a.status;
  $("appointment-editor").open = true;
  $("appointment-title").focus();
}

bindForm("create-profile", "create-error", async () => {
  const data = await api("profiles", "POST", {
    name: value("create-name"),
    caregiver: value("create-caregiver"),
    email: value("create-email"),
  });
  saveSession(data);
  await load();
  $("share-id").textContent = data.profile.id;
  $("share-code").textContent = data.access_code;
  $("share-profile").hidden = false;
  $("profile-heading").tabIndex = -1;
  $("profile-heading").focus();
});
bindForm("join-profile", "join-error", async () => {
  saveSession(
    await api("login", "POST", {
      profile_id: value("join-id"),
      access_code: value("join-code"),
      caregiver: value("join-caregiver"),
    }),
  );
  $("join-code").value = "";
  await load();
  $("profile-heading").tabIndex = -1;
  $("profile-heading").focus();
});
bindForm("edit-profile", "profile-error", async () => {
  await api("profile", "PUT", {
    name: value("profile-name"),
    allergies: value("profile-allergies"),
    care_note: value("profile-note"),
    email: value("profile-email"),
    timezone: value("profile-zone"),
    reminders_enabled: $("profile-reminders").checked,
    version: state.profileFormVersion,
  });
  state.profileDirty = false;
  await load();
});
bindForm("medication-form", "medication-error", async () => {
  await api(
    `medications${state.medEdit ? "/" + state.medEdit.id : ""}`,
    state.medEdit ? "PUT" : "POST",
    medBody(),
  );
  state.medEdit = null;
  $("medication-form").reset();
  $("medication-editor").open = false;
  await load();
});
bindForm("appointment-form", "appointment-error", async () => {
  await api(
    `appointments${state.appointmentEdit ? "/" + state.appointmentEdit.id : ""}`,
    state.appointmentEdit ? "PUT" : "POST",
    appointmentBody(),
  );
  state.appointmentEdit = null;
  $("appointment-form").reset();
  $("appointment-editor").open = false;
  await load();
});
$("cancel-medication-edit").onclick = () => {
  state.medEdit = null;
  $("medication-form").reset();
  $("medication-start").value = localParts().day;
  $("medication-editor").open = false;
};
$("cancel-appointment-edit").onclick = () => {
  state.appointmentEdit = null;
  $("appointment-form").reset();
  $("appointment-editor").open = false;
};
$("refresh-plan").onclick = () =>
  load()
    .then(() => announce(t("saved")))
    .catch((e) => announce(e.message, true));
$("print-plan").onclick = () => window.print();
$("care-logout").onclick = async () => {
  try {
    await api("logout", "POST");
  } finally {
    session = null;
    sessionStorage.removeItem("behaviorsense.care-session");
    state.profile = null;
    $("share-code").textContent = "";
    $("share-profile").hidden = true;
    $("care-workspace").hidden = true;
    $("care-auth").hidden = false;
    announce(t("signedOut"));
    $("join-id").focus();
  }
};
$("copy-access").onclick = async () => {
  try {
    await navigator.clipboard.writeText(
      `${t("profileId")}: ${$("share-id").textContent}\n${t("accessCode")}: ${$("share-code").textContent}`,
    );
    announce(t("copied"));
  } catch {
    announce(
      "Select and copy the profile ID and access code shown above.",
      true,
    );
  }
};
$("hide-access").onclick = () => {
  $("share-code").textContent = "";
  $("share-profile").hidden = true;
};
$("check-reminders").onclick = async () => {
  try {
    await api("reminders/check", "POST");
    await load();
    announce(t("saved"));
  } catch (e) {
    announce(e.message, true);
  }
};
$("edit-profile").addEventListener("input", () => {
  state.profileDirty = true;
});
$("care-language").onchange = () => {
  language = $("care-language").value;
  try {
    localStorage.setItem("behaviorsense.care-language", language);
  } catch {}
  translations();
  render();
};
$("care-workspace").addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const { action, id, time, day } = button.dataset;
  if (action === "edit-medication") {
    editMedication(state.medications.find((m) => m.id === id));
    return;
  }
  if (action === "edit-appointment") {
    editAppointment(state.appointments.find((a) => a.id === id));
    return;
  }
  button.disabled = true;
  try {
    if (["taken", "skipped"].includes(action))
      await api(`medications/${id}/confirm`, "POST", {
        day,
        time,
        status: action,
        note:
          button
            .closest(".care-item")
            .querySelector("[data-dose-note]")
            ?.value.trim() || "",
      });
    if (action === "stop-medication") await api(`medications/${id}`, "DELETE");
    if (action === "cancel-appointment")
      await api(`appointments/${id}`, "DELETE");
    if (action === "attended") {
      const a = state.appointments.find((a) => a.id === id);
      await api(`appointments/${id}`, "PUT", {
        title: a.title,
        doctor: a.doctor,
        local_at: a.local_at,
        location: a.location,
        notes: a.notes,
        status: "attended",
        version: a.version,
      });
    }
    await load();
    const heading = $(
      ["taken", "skipped"].includes(action)
        ? "today-heading"
        : action === "stop-medication"
          ? "medication-heading"
          : "appointment-heading",
    );
    heading.tabIndex = -1;
    heading.focus({ preventScroll: true });
    announce(t("saved"));
  } catch (e) {
    announce(e.message, true);
    button.disabled = false;
  }
});
$("care-demo").onclick = async () => {
  const button = $("care-demo");
  button.disabled = true;
  try {
    const health = await api("health");
    saveSession(
      await api("profiles", "POST", {
        name: "Demo Resident",
        caregiver: "Demo Caregiver",
        email:
          health.mail_mode === "preview" ? "care-demo@example.invalid" : "",
        reminders_enabled: health.mail_mode === "preview",
        allergies: "Fictional example — no real patient information.",
        care_note:
          "This profile is a demonstration. Enter clinician-provided details for an actual care plan.",
      }),
    );
    const upcoming = localParts();
    await api("medications", "POST", {
      name: "Prescribed medicine (demo)",
      dose: "Clinician-provided dose (demo)",
      instructions: "Fictional example for the care diary.",
      times: [upcoming.time],
      start_day: upcoming.day,
      active: true,
    });
    const visit = localParts(new Date(Date.now() + 45 * 60000));
    await api("appointments", "POST", {
      title: "Care review (demo)",
      doctor: "Demo care professional",
      local_at: `${visit.day}T${visit.time}`,
      location: "Fictional example",
      notes: "No clinic booking is made.",
      status: "scheduled",
    });
    await api("reminders/check", "POST");
    await load();
    announce(t("demoNotice"));
    $("profile-heading").tabIndex = -1;
    $("profile-heading").focus();
  } catch (e) {
    announce(e.message, true);
  } finally {
    button.disabled = false;
  }
};
translations();
if (session) load().catch((e) => announce(e.message, true));
else api("health").catch(() => announce(t("offline"), true));
