import {mountAccessibility} from "./accessibility.js";
import {summarizeDoses} from "./care-summary.js";

const $=id=>document.getElementById(id);
const escape=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const WORDS={
 skip:["Skip to the care plan","देखभाल योजना पर जाएँ","काळजी योजनेकडे जा"],
 vertical:["Elderly care & assisted living","बुज़ुर्गों की देखभाल","ज्येष्ठांची काळजी"],
 headline:["One shared plan for everyday care.","रोज़ की देखभाल के लिए एक साझा योजना।","दैनंदिन काळजीसाठी एक सामायिक योजना."],
 intro:["Keep prescribed medication, caregiver confirmations and doctor appointments together. Share access with people you trust.","दवाइयों, देखभाल की पुष्टि और डॉक्टर की मुलाकातों को एक साथ रखें। भरोसेमंद लोगों के साथ पहुँच साझा करें।","औषधे, काळजीवाहकांची पुष्टी आणि डॉक्टरांच्या भेटी एकत्र ठेवा. विश्वासू व्यक्तींना प्रवेश द्या."],
 scope:["Intake is recorded by a caregiver. Video activity predictions stay separate from medication confirmations.","दवा लेने की पुष्टि देखभाल करने वाला दर्ज करता है। वीडियो की गतिविधियाँ इससे अलग रहती हैं।","औषध घेतल्याची पुष्टी काळजीवाहक नोंदवतो. व्हिडिओतील हालचालींचे अंदाज स्वतंत्र राहतात."],
 today:["Today","आज","आज"],medications:["Medications","दवाइयाँ","औषधे"],appointments:["Appointments","मुलाकातें","भेटी"],observations:["Activity observations","गतिविधि अवलोकन","हालचालींचे निरीक्षण"],language:["Language","भाषा","भाषा"],
 create:["Create a care profile","देखभाल प्रोफ़ाइल बनाएँ","काळजी प्रोफाइल तयार करा"],join:["Open a shared care profile","साझा प्रोफ़ाइल खोलें","सामायिक प्रोफाइल उघडा"],
 personName:["Person receiving care","देखभाल पाने वाले का नाम","काळजी घेण्यात येणाऱ्या व्यक्तीचे नाव"],caregiverName:["Your caregiver name","देखभाल करने वाले के रूप में आपका नाम","काळजीवाहक म्हणून तुमचे नाव"],
 email:["Reminder email (optional)","याद दिलाने के लिए ईमेल (वैकल्पिक)","स्मरणपत्रासाठी ईमेल (ऐच्छिक)"],
 codeGenerated:["A secure access code is generated. Save the profile ID and code to share with your care team.","सुरक्षित पहुँच कोड बनेगा। प्रोफ़ाइल आईडी और कोड सुरक्षित रखकर अपनी टीम से साझा करें।","सुरक्षित प्रवेश कोड तयार होईल. प्रोफाइल आयडी आणि कोड जतन करून काळजी संघाशी शेअर करा."],
 profileId:["Profile ID","प्रोफ़ाइल आईडी","प्रोफाइल आयडी"],accessCode:["Access code","पहुँच कोड","प्रवेश कोड"],
 joinHelp:["Ask a trusted care team member for the profile ID and access code.","भरोसेमंद देखभाल सदस्य से प्रोफ़ाइल आईडी और पहुँच कोड लें।","विश्वासू काळजी संघ सदस्याकडून प्रोफाइल आयडी आणि प्रवेश कोड घ्या."],
 demo:["Try a fictional care profile","काल्पनिक प्रोफ़ाइल आज़माएँ","काल्पनिक प्रोफाइल वापरून पहा"],
 demoGuide:["Try the care workflow in two minutes","दो मिनट में देखभाल योजना आज़माएँ","दोन मिनिटांत काळजी योजना वापरून पहा"],
 demoStep1:["Choose “Try a fictional care profile” below. It creates a private example with a dose due now and an appointment.","नीचे काल्पनिक प्रोफ़ाइल का विकल्प चुनें। अभी की दवा और एक मुलाकात का निजी उदाहरण बनेगा।","खाली काल्पनिक प्रोफाइलचा पर्याय निवडा. आताची औषधाची वेळ आणि एका भेटीचे खाजगी उदाहरण तयार होईल."],
 demoStep2:["Record a taken or skipped confirmation, with an optional note. Check today's counts and the history.","दवा ली या नहीं ली की पुष्टि और चाहें तो टिप्पणी दर्ज करें। आज की गिनती और इतिहास देखें।","औषध घेतले किंवा वगळले याची पुष्टी आणि हवी असल्यास नोंद करा. आजची संख्या आणि इतिहास पहा."],
 demoStep3:["Add or edit a prescribed medication, then update the doctor appointment.","निर्धारित दवा जोड़ें या बदलें, फिर डॉक्टर की मुलाकात बदलें।","लिहून दिलेले औषध जोडा किंवा बदला, नंतर डॉक्टरांची भेट अद्ययावत करा."],
 demoStep4:["Review email reminder previews, switch languages, and try the large-text or high-contrast controls.","ईमेल स्मरण का पूर्वावलोकन देखें, भाषा बदलें और बड़े अक्षर या अधिक कंट्रास्ट आज़माएँ।","ईमेल स्मरणपत्रांचे पूर्वावलोकन पहा, भाषा बदला आणि मोठा मजकूर किंवा जास्त रंग विरोध वापरून पहा."],
 demoStep5:["Print the care plan for a visit. Open Activity observations to explore the optional evidence-checked video reports.","मुलाकात के लिए योजना प्रिंट करें। वीडियो रिपोर्ट के लिए गतिविधि अवलोकन खोलें।","भेटीसाठी काळजी योजना छापा. व्हिडिओ अहवाल पाहण्यासाठी हालचालींचे निरीक्षण उघडा."],
 printPlan:["Print care plan","देखभाल योजना प्रिंट करें","काळजी योजना छापा"],
 confirmedTotal:["caregiver confirmations out of today's scheduled doses","आज की निर्धारित दवाओं में देखभाल की पुष्टि","आजच्या नियोजित औषधांपैकी काळजीवाहकांच्या पुष्टी"],
 summaryHelp:["Counts include later doses today. Pending or overdue means no confirmation is recorded.","गिनती में आज की बाद की दवाएँ भी हैं। लंबित या देरी का मतलब पुष्टि दर्ज नहीं है।","आजच्या पुढील औषधांचाही संख्येत समावेश आहे. प्रलंबित किंवा उशीर म्हणजे पुष्टी नोंदवलेली नाही."],
 confirmationNote:["Optional confirmation note","पुष्टि की टिप्पणी (वैकल्पिक)","पुष्टीची नोंद (ऐच्छिक)"],
 note:["Caregiver note","देखभाल की टिप्पणी","काळजीवाहकाची नोंद"],
 refresh:["Refresh plan","योजना फिर लोड करें","योजना पुन्हा उघडा"],logout:["Sign out","साइन आउट","बाहेर पडा"],
 saveAccess:["Save this access information","यह पहुँच जानकारी सुरक्षित रखें","प्रवेश माहिती जतन करा"],
 shareHelp:["Share these details privately with trusted caregivers. The code is shown only when the profile is created.","जानकारी केवल भरोसेमंद देखभाल सदस्यों से निजी रूप में साझा करें। कोड प्रोफ़ाइल बनाते समय दिखता है।","ही माहिती विश्वासू काळजीवाहकांशी खाजगीपणे शेअर करा. कोड प्रोफाइल तयार करताना दिसतो."],
 copyAccess:["Copy access information","पहुँच जानकारी कॉपी करें","प्रवेश माहिती कॉपी करा"],hideAccess:["Hide access code","पहुँच कोड छिपाएँ","प्रवेश कोड लपवा"],
 editProfile:["Edit care profile and reminders","प्रोफ़ाइल और याद दिलाने की सुविधा बदलें","प्रोफाइल आणि स्मरणपत्रे बदला"],
 allergies:["Allergies / important health information","एलर्जी / महत्वपूर्ण स्वास्थ्य जानकारी","अॅलर्जी / महत्त्वाची आरोग्य माहिती"],careNotes:["Shared care notes","साझा देखभाल नोट्स","सामायिक काळजी नोंदी"],timezone:["Care plan time zone","योजना का समय क्षेत्र","योजनेचा वेळ विभाग"],
 timeHelp:["All medication and appointment times use this care plan time zone.","दवाइयों और मुलाकातों का समय इस योजना के समय क्षेत्र में है।","औषधे आणि भेटींचे वेळापत्रक या वेळ विभागानुसार आहे."],
 enableReminders:["Enable email reminders","ईमेल याद दिलाना चालू करें","ईमेल स्मरणपत्रे सुरू करा"],saveProfile:["Save care profile","प्रोफ़ाइल सहेजें","प्रोफाइल जतन करा"],
 todayDoses:["Today's medication confirmations","आज की दवा लेने की पुष्टि","आजच्या औषधांची पुष्टी"],
 confirmationHelp:["Confirm only what you or the caregiver know. An overdue confirmation does not establish whether medication was taken.","केवल ज्ञात जानकारी की पुष्टि करें। देरी से दवा न लेने का प्रमाण नहीं मिलता।","तुम्हाला किंवा काळजीवाहकाला माहीत असलेली माहितीच नोंदवा. उशीर झाला म्हणजे औषध घेतले नाही असा पुरावा होत नाही."],
 addMedication:["Add prescribed medication","निर्धारित दवा जोड़ें","लिहून दिलेले औषध जोडा"],medicationName:["Medication name","दवा का नाम","औषधाचे नाव"],prescribedDose:["Prescribed dose","निर्धारित मात्रा","लिहून दिलेली मात्रा"],doseTimes:["Daily times (24-hour, comma-separated)","रोज़ का समय (24 घंटे, कॉमा से अलग)","दैनंदिन वेळा (24 तास, स्वल्पविरामाने वेगळ्या)"],startDate:["Start date","शुरू होने की तारीख","सुरू करण्याची तारीख"],instructions:["Instructions from the care plan","देखभाल योजना के निर्देश","काळजी योजनेतील सूचना"],activeMedication:["Medication is active","दवा चालू है","औषध सुरू आहे"],
 prescriptionHelp:["Enter the medication and dose supplied by the clinician. This app does not prescribe or change doses.","डॉक्टर की बताई दवा और मात्रा दर्ज करें। ऐप दवा या मात्रा तय नहीं करता।","डॉक्टरांनी सांगितलेले औषध आणि मात्रा नोंदवा. अॅप औषध किंवा मात्रा ठरवत नाही."],
 saveMedication:["Save medication","दवा सहेजें","औषध जतन करा"],cancelEdit:["Cancel edit","बदलना रद्द करें","बदल रद्द करा"],addAppointment:["Add doctor appointment","डॉक्टर की मुलाकात जोड़ें","डॉक्टरांची भेट जोडा"],appointmentTitle:["Appointment / reason","मुलाकात / कारण","भेट / कारण"],doctor:["Doctor / care professional","डॉक्टर / देखभाल विशेषज्ञ","डॉक्टर / काळजी तज्ज्ञ"],appointmentTime:["Date and time in the care plan time zone","योजना के समय क्षेत्र में तारीख और समय","योजनेच्या वेळ विभागातील तारीख आणि वेळ"],location:["Location / visit link","स्थान / मुलाकात लिंक","ठिकाण / भेटीची लिंक"],appointmentStatus:["Appointment status","मुलाकात की स्थिति","भेटीची स्थिती"],scheduled:["Scheduled","तय है","नियोजित"],attended:["Attended","मुलाकात हुई","भेट झाली"],cancelled:["Cancelled","रद्द","रद्द"],appointmentHelp:["This records your care team's appointment. It does not book a visit with a clinic.","यह आपकी टीम की मुलाकात का रिकॉर्ड है। क्लिनिक में बुकिंग अलग से करें।","ही तुमच्या संघाच्या भेटीची नोंद आहे. दवाखान्यातील आरक्षण स्वतंत्रपणे करा."],saveAppointment:["Save appointment","मुलाकात सहेजें","भेट जतन करा"],
 reminders:["Email reminders","ईमेल स्मरण","ईमेल स्मरणपत्रे"],checkReminders:["Check due reminders","बकाया स्मरण जाँचें","देय स्मरणपत्रे तपासा"],history:["Caregiver confirmation history","देखभाल पुष्टि का इतिहास","काळजीवाहक पुष्टीचा इतिहास"],footer:["Care records stay in your care server's private database. Medication confirmations are entered by caregivers.","रिकॉर्ड आपके देखभाल सर्वर के निजी डेटाबेस में रहते हैं। दवा की पुष्टि देखभाल सदस्य दर्ज करते हैं।","नोंदी तुमच्या काळजी सर्व्हरच्या खाजगी डेटाबेसमध्ये राहतात. औषधांची पुष्टी काळजीवाहक नोंदवतात."],
 pending:["Awaiting confirmation","पुष्टि की प्रतीक्षा","पुष्टीची प्रतीक्षा"],overdue:["Overdue confirmation","पुष्टि में देरी","पुष्टी उशिरा आहे"],taken:["Confirmed taken","दवा लेने की पुष्टि","औषध घेतल्याची पुष्टी"],skipped:["Reported skipped","दवा छोड़ना दर्ज","औषध वगळल्याची नोंद"],
 confirmTaken:["Confirm taken","दवा लेने की पुष्टि करें","औषध घेतल्याची पुष्टी करा"],confirmSkipped:["Report skipped","दवा छोड़ना दर्ज करें","औषध वगळल्याची नोंद करा"],edit:["Edit","बदलें","बदला"],stop:["Stop in care plan","योजना में बंद करें","योजनेत थांबवा"],markAttended:["Mark attended","मुलाकात हुई दर्ज करें","भेट झाली नोंदवा"],cancelAppointment:["Cancel in care plan","योजना में रद्द करें","योजनेत रद्द करा"],
 noDoses:["No active doses are scheduled for this day.","इस दिन के लिए दवा का समय दर्ज नहीं है।","या दिवसासाठी औषधांचे वेळापत्रक नाही."],noMedications:["Add the medications prescribed for this person.","इस व्यक्ति के लिए निर्धारित दवाइयाँ जोड़ें।","या व्यक्तीसाठी लिहून दिलेली औषधे जोडा."],noAppointments:["No appointments have been added.","कोई मुलाकात दर्ज नहीं है।","अजून भेटी नोंदवलेल्या नाहीत."],noReminders:["No reminder messages are queued. Enable reminders in the profile and check when a dose or appointment is due.","स्मरण संदेश नहीं हैं। प्रोफ़ाइल में स्मरण चालू करें और तय समय पर जाँचें।","स्मरण संदेश नाहीत. प्रोफाइलमध्ये स्मरणपत्रे सुरू करा आणि वेळ जवळ आल्यावर तपासा."],noHistory:["Caregiver confirmations will appear here.","देखभाल की पुष्टि यहाँ दिखेगी।","काळजीवाहकांची पुष्टी येथे दिसेल."],
 preview:["Preview","पूर्वावलोकन","पूर्वावलोकन"],queued:["Queued","प्रतीक्षा में","रांगेत"],sent:["Sent","भेजा गया","पाठवले"],failed:["Delivery failed","भेजना विफल","पाठवणे अयशस्वी"],sending:["Sending","भेजा जा रहा है","पाठवत आहे"],
 previewHelp:["Email previews are stored locally. SMTP delivery can be enabled in the server configuration.","ईमेल का पूर्वावलोकन स्थानीय रूप में सुरक्षित है। भेजने के लिए सर्वर में SMTP चालू करें।","ईमेलचे पूर्वावलोकन स्थानिकपणे जतन होते. पाठवण्यासाठी सर्व्हरमध्ये SMTP सुरू करा."],smtpHelp:["Email delivery is configured. Reminders are sent only for profiles with email reminders enabled.","ईमेल भेजना चालू है। केवल स्मरण चालू प्रोफ़ाइल को संदेश भेजे जाएँगे।","ईमेल पाठवणे सुरू आहे. स्मरणपत्रे चालू असलेल्या प्रोफाइलनाच संदेश पाठवले जातील."],
 loading:["Loading care plan…","योजना लोड हो रही है…","योजना उघडत आहे…"],saved:["Care plan saved.","देखभाल योजना सहेजी गई।","काळजी योजना जतन झाली."],signedOut:["You have signed out.","आप साइन आउट हो गए।","तुम्ही बाहेर पडलात."],copied:["Access information copied. Share it privately.","पहुँच जानकारी कॉपी हुई। निजी रूप से साझा करें।","प्रवेश माहिती कॉपी झाली. खाजगीपणे शेअर करा."],error:["Please check the information and try again.","जानकारी जाँचें और फिर कोशिश करें।","माहिती तपासा आणि पुन्हा प्रयत्न करा."],offline:["The care service is unavailable. Open the app from the running care server.","देखभाल सेवा उपलब्ध नहीं है। चल रहे देखभाल सर्वर से ऐप खोलें।","काळजी सेवा उपलब्ध नाही. सुरू असलेल्या काळजी सर्व्हरवरून अॅप उघडा."],
 carePlan:["care plan","देखभाल योजना","काळजी योजना"],day:["Day","दिन","दिवस"],medication:["Medication","दवा","औषध"],dose:["Dose","मात्रा","मात्रा"],status:["Status","स्थिति","स्थिती"],reportedBy:["Reported by","दर्ज करने वाला","नोंद करणारा"],recordedAt:["Recorded at","दर्ज करने का समय","नोंद करण्याची वेळ"],inactive:["Stopped","बंद","थांबवले"],dueLater:["Confirmation opens 15 minutes before the dose is due.","तय समय से 15 मिनट पहले पुष्टि खुलेगी।","औषधाच्या वेळेच्या 15 मिनिटे आधी पुष्टी करता येईल."],
 demoNotice:["Fictional demo profile created. All sample records are examples.","काल्पनिक नमूना प्रोफ़ाइल बना। सभी रिकॉर्ड उदाहरण हैं।","काल्पनिक नमुना प्रोफाइल तयार झाला. सर्व नोंदी उदाहरणे आहेत."]
};
let language="en";
try {language=localStorage.getItem("behaviorsense.care-language")||"en"}catch{}
if(!["en","hi","mr"].includes(language))language="en";
const t=key=>(WORDS[key]||[key,key,key])[["en","hi","mr"].indexOf(language)];
let session=null;
try{session=JSON.parse(sessionStorage.getItem("behaviorsense.care-session")||"null")}catch{}
const state={profile:null,medications:[],doses:[],appointments:[],outbox:[],history:[],mode:"preview",medEdit:null,appointmentEdit:null,profileDirty:false,profileFormVersion:null};
const value=id=>$(id).value.trim();
function announce(message,error=false){$("care-status").textContent=message;$("care-status").classList.toggle("error",error);$("care-status").setAttribute("aria-live",error?"assertive":"polite");}
function localParts(when=new Date()){
 const parts=new Intl.DateTimeFormat("en-CA",{timeZone:state.profile?.timezone||"Asia/Kolkata",year:"numeric",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",hourCycle:"h23"}).formatToParts(when);
 const p=Object.fromEntries(parts.map(x=>[x.type,x.value]));return{day:`${p.year}-${p.month}-${p.day}`,time:`${p.hour}:${p.minute}`};
}
function when(iso){return new Intl.DateTimeFormat({en:"en-IN",hi:"hi-IN",mr:"mr-IN"}[language],{timeZone:state.profile?.timezone||"Asia/Kolkata",dateStyle:"medium",timeStyle:"short"}).format(new Date(iso));}
async function api(path,method="GET",body){
 const response=await fetch(`/api/care/${path}`,{method,headers:{...(session?{Authorization:`Bearer ${session.token}`} :{}),...(body?{"Content-Type":"application/json"}:{})},...(body?{body:JSON.stringify(body)}:{})});
 if(response.status===204)return null;
 const data=await response.json();
 if(!response.ok){if(response.status===401){session=null;sessionStorage.removeItem("behaviorsense.care-session");$("care-auth").hidden=false;$("care-workspace").hidden=true;}const detail=Array.isArray(data.detail)?data.detail.map(item=>item.msg).join("; "):data.detail;throw new Error(detail||t("error"));}
 return data;
}
function saveSession(data){session={token:data.token,caregiver:data.caregiver};sessionStorage.setItem("behaviorsense.care-session",JSON.stringify(session));state.profile=data.profile;}
function translations(){document.documentElement.lang=language;$("care-language").value=language;for(const el of document.querySelectorAll("[data-care-text]"))el.textContent=t(el.dataset.careText);mountAccessibility(language);}
function badge(status){return `<span class="care-badge ${escape(status)}">${escape(t(status))}</span>`;}
function render(){
 const p=state.profile;if(!p)return;
 $("care-auth").hidden=true;$("care-workspace").hidden=false;
 $("profile-heading").textContent=`${p.name} · ${t("carePlan")}`;$("profile-meta").textContent=`${t("profileId")}: ${p.id} · ${p.timezone} · ${session.caregiver}`;
 if(!state.profileDirty){$("profile-name").value=p.name;$("profile-email").value=p.email;$("profile-allergies").value=p.allergies;$("profile-note").value=p.care_note;$("profile-zone").value=p.timezone;$("profile-reminders").checked=p.reminders_enabled;state.profileFormVersion=p.version;}
 $("care-day").textContent=`${localParts().day} · ${p.timezone}`;
 const counts=summarizeDoses(state.doses);
 $("daily-confirmations").innerHTML=`<p><strong>${counts.confirmed} / ${counts.scheduled}</strong> ${escape(t("confirmedTotal"))}</p><dl>${["taken","skipped","pending","overdue"].map(key=>`<div><dt>${escape(t(key))}</dt><dd data-dose-count="${key}">${counts[key]}</dd></div>`).join("")}</dl><p class="care-help">${escape(t("summaryHelp"))}</p>`;
 $("dose-list").innerHTML=state.doses.length?state.doses.map(d=>{
  const confirmed=["taken","skipped"].includes(d.status),early=new Date(d.due_at).getTime()>Date.now()+900000;
  const noteId=`dose-note-${d.medication_id}-${d.time.replace(":","")}`;
  const note=confirmed&&d.note?`<p>${escape(t("note"))}: ${escape(d.note)}</p>`:!confirmed&&!early?`<label for="${noteId}">${escape(t("confirmationNote"))}: ${escape(d.name)}, ${escape(d.time)}</label><textarea id="${noteId}" data-dose-note maxlength="600" rows="2"></textarea>`:"";
  return `<article class="care-item"><div><h3>${escape(d.name)}</h3><p>${escape(d.dose)} · ${escape(d.time)} (${escape(p.timezone)})</p><p>${escape(d.instructions)}</p>${badge(d.status)}${confirmed?`<p>${escape(t("reportedBy"))}: ${escape(d.caregiver)} · ${escape(when(d.confirmed_at))}</p>`:""}${note}${early?`<p class="care-help">${escape(t("dueLater"))}</p>`:""}</div>${!confirmed?`<div class="care-actions"><button type="button" data-action="taken" data-id="${d.medication_id}" data-time="${d.time}" data-day="${d.day}" ${early?"disabled":""} aria-label="${escape(t("confirmTaken"))}: ${escape(d.name)}, ${d.time}">${escape(t("confirmTaken"))}</button><button type="button" class="secondary" data-action="skipped" data-id="${d.medication_id}" data-time="${d.time}" data-day="${d.day}" ${early?"disabled":""} aria-label="${escape(t("confirmSkipped"))}: ${escape(d.name)}, ${d.time}">${escape(t("confirmSkipped"))}</button></div>`:""}</article>`;
 }).join(""):`<p class="empty">${escape(t("noDoses"))}</p>`;
 $("medication-list").innerHTML=state.medications.length?state.medications.map(m=>`<article class="care-item"><div><h3>${escape(m.name)}</h3><p>${escape(m.dose)} · ${escape(m.times.join(", "))}</p><p>${escape(m.instructions)}</p>${!m.active?badge("inactive"):""}</div><div class="care-actions"><button type="button" class="secondary" data-action="edit-medication" data-id="${m.id}" aria-label="${escape(t("edit"))}: ${escape(m.name)}">${escape(t("edit"))}</button>${m.active?`<button type="button" class="secondary" data-action="stop-medication" data-id="${m.id}" aria-label="${escape(t("stop"))}: ${escape(m.name)}">${escape(t("stop"))}</button>`:""}</div></article>`).join(""):`<p class="empty">${escape(t("noMedications"))}</p>`;
 $("appointment-list").innerHTML=state.appointments.length?state.appointments.map(a=>`<article class="care-item"><div><h3>${escape(a.title)}</h3><p>${escape(a.doctor)} · ${escape(when(a.due_at))}</p><p>${escape(a.location)}</p><p>${escape(a.notes)}</p>${badge(a.status)}</div><div class="care-actions"><button type="button" class="secondary" data-action="edit-appointment" data-id="${a.id}" aria-label="${escape(t("edit"))}: ${escape(a.title)}">${escape(t("edit"))}</button>${a.status==="scheduled"?`<button type="button" data-action="attended" data-id="${a.id}">${escape(t("markAttended"))}</button><button type="button" class="secondary" data-action="cancel-appointment" data-id="${a.id}">${escape(t("cancelAppointment"))}</button>`:""}</div></article>`).join(""):`<p class="empty">${escape(t("noAppointments"))}</p>`;
 $("reminder-mode").textContent=t(state.mode==="smtp"?"smtpHelp":"previewHelp");
 $("reminder-list").innerHTML=state.outbox.length?state.outbox.map(m=>`<article class="care-item"><div><h3>${escape(m.subject)}</h3><p>${escape(m.recipient)} · ${escape(when(m.created_at))}</p>${badge(m.status)}<details><summary>${escape(t("preview"))}</summary><p>${escape(m.body).replace(/\n/g,"<br>")}</p>${m.error?`<p>${escape(m.error)}</p>`:""}</details></div></article>`).join(""):`<p class="empty">${escape(t("noReminders"))}</p>`;
 $("care-history").innerHTML=state.history.length?`<div class="care-table-wrap" tabindex="0" role="region" aria-label="${escape(t("history"))}"><table><caption>${escape(t("history"))}</caption><thead><tr>${["day","medication","dose","status","reportedBy","recordedAt","note"].map(key=>`<th scope="col">${escape(t(key))}</th>`).join("")}</tr></thead><tbody>${state.history.map(r=>`<tr><td>${escape(r.day)} ${escape(r.time)}</td><td>${escape(r.medication_name)}</td><td>${escape(r.dose)}</td><td>${escape(t(r.status))}</td><td>${escape(r.caregiver)}</td><td>${escape(when(r.confirmed_at))}</td><td>${escape(r.note)}</td></tr>`).join("")}</tbody></table></div>`:`<p class="empty">${escape(t("noHistory"))}</p>`;
 if(!state.medEdit)$("medication-start").value=localParts().day;
 translations();
}
async function load(){
 const [profile,medications,doses,appointments,reminders,history]=await Promise.all([api("profile"),api("medications"),api("doses"),api("appointments"),api("reminders"),api("history")]);
 Object.assign(state,{profile,medications,doses,appointments,outbox:reminders.messages,mode:reminders.mode,history});render();
}
function errorFor(id,error){const el=$(id);el.textContent=`${t("error")} ${error.message}`;el.hidden=false;el.focus();}
function bindForm(id,errorId,handler){$(id).addEventListener("submit",async event=>{event.preventDefault();const button=$(id).querySelector('button[type="submit"]');button.disabled=true;$(errorId).hidden=true;try{await handler();announce(t("saved"));}catch(error){errorFor(errorId,error);}finally{button.disabled=false;}});}
function medBody(){return{name:value("medication-name"),dose:value("medication-dose"),instructions:value("medication-instructions"),times:value("medication-times").split(",").map(s=>s.trim()).filter(Boolean),start_day:value("medication-start"),active:$("medication-active").checked,...(state.medEdit?{version:state.medEdit.version}:{})};}
function appointmentBody(){return{title:value("appointment-title"),doctor:value("appointment-doctor"),local_at:value("appointment-time"),location:value("appointment-location"),notes:value("appointment-notes"),status:value("appointment-status"),...(state.appointmentEdit?{version:state.appointmentEdit.version}:{})};}
function editMedication(m){state.medEdit=m;$("medication-name").value=m.name;$("medication-dose").value=m.dose;$("medication-times").value=m.times.join(", ");$("medication-start").value=m.start_day;$("medication-instructions").value=m.instructions;$("medication-active").checked=m.active;$("medication-editor").open=true;$("medication-name").focus();}
function editAppointment(a){state.appointmentEdit=a;$("appointment-title").value=a.title;$("appointment-doctor").value=a.doctor;$("appointment-time").value=a.local_at;$("appointment-location").value=a.location;$("appointment-notes").value=a.notes;$("appointment-status").value=a.status;$("appointment-editor").open=true;$("appointment-title").focus();}

bindForm("create-profile","create-error",async()=>{const data=await api("profiles","POST",{name:value("create-name"),caregiver:value("create-caregiver"),email:value("create-email")});saveSession(data);await load();$("share-id").textContent=data.profile.id;$("share-code").textContent=data.access_code;$("share-profile").hidden=false;$("profile-heading").tabIndex=-1;$("profile-heading").focus();});
bindForm("join-profile","join-error",async()=>{saveSession(await api("login","POST",{profile_id:value("join-id"),access_code:value("join-code"),caregiver:value("join-caregiver")}));$("join-code").value="";await load();$("profile-heading").tabIndex=-1;$("profile-heading").focus();});
bindForm("edit-profile","profile-error",async()=>{await api("profile","PUT",{name:value("profile-name"),allergies:value("profile-allergies"),care_note:value("profile-note"),email:value("profile-email"),timezone:value("profile-zone"),reminders_enabled:$("profile-reminders").checked,version:state.profileFormVersion});state.profileDirty=false;await load();});
bindForm("medication-form","medication-error",async()=>{await api(`medications${state.medEdit?"/"+state.medEdit.id:""}`,state.medEdit?"PUT":"POST",medBody());state.medEdit=null;$("medication-form").reset();$("medication-editor").open=false;await load();});
bindForm("appointment-form","appointment-error",async()=>{await api(`appointments${state.appointmentEdit?"/"+state.appointmentEdit.id:""}`,state.appointmentEdit?"PUT":"POST",appointmentBody());state.appointmentEdit=null;$("appointment-form").reset();$("appointment-editor").open=false;await load();});
$("cancel-medication-edit").onclick=()=>{state.medEdit=null;$("medication-form").reset();$("medication-start").value=localParts().day;$("medication-editor").open=false;};
$("cancel-appointment-edit").onclick=()=>{state.appointmentEdit=null;$("appointment-form").reset();$("appointment-editor").open=false;};
$("refresh-plan").onclick=()=>load().then(()=>announce(t("saved"))).catch(e=>announce(e.message,true));
$("print-plan").onclick=()=>window.print();
$("care-logout").onclick=async()=>{try{await api("logout","POST")}finally{session=null;sessionStorage.removeItem("behaviorsense.care-session");state.profile=null;$("share-code").textContent="";$("share-profile").hidden=true;$("care-workspace").hidden=true;$("care-auth").hidden=false;announce(t("signedOut"));$("join-id").focus();}};
$("copy-access").onclick=async()=>{try{await navigator.clipboard.writeText(`${t("profileId")}: ${$("share-id").textContent}\n${t("accessCode")}: ${$("share-code").textContent}`);announce(t("copied"));}catch{announce("Select and copy the profile ID and access code shown above.",true);}};
$("hide-access").onclick=()=>{$("share-code").textContent="";$("share-profile").hidden=true;};
$("check-reminders").onclick=async()=>{try{await api("reminders/check","POST");await load();announce(t("saved"));}catch(e){announce(e.message,true);}};
$("edit-profile").addEventListener("input",()=>{state.profileDirty=true;});
$("care-language").onchange=()=>{language=$("care-language").value;try{localStorage.setItem("behaviorsense.care-language",language)}catch{}translations();render();};
$("care-workspace").addEventListener("click",async event=>{
 const button=event.target.closest("button[data-action]");if(!button)return;
 const {action,id,time,day}=button.dataset;
 if(action==="edit-medication"){editMedication(state.medications.find(m=>m.id===id));return;}
 if(action==="edit-appointment"){editAppointment(state.appointments.find(a=>a.id===id));return;}
 button.disabled=true;
 try{
  if(["taken","skipped"].includes(action))await api(`medications/${id}/confirm`,"POST",{day,time,status:action,note:button.closest(".care-item").querySelector("[data-dose-note]")?.value.trim()||""});
  if(action==="stop-medication")await api(`medications/${id}`,"DELETE");
  if(action==="cancel-appointment")await api(`appointments/${id}`,"DELETE");
  if(action==="attended"){const a=state.appointments.find(a=>a.id===id);await api(`appointments/${id}`,"PUT",{title:a.title,doctor:a.doctor,local_at:a.local_at,location:a.location,notes:a.notes,status:"attended",version:a.version});}
  await load();const heading=$(["taken","skipped"].includes(action)?"today-heading":action==="stop-medication"?"medication-heading":"appointment-heading");heading.tabIndex=-1;heading.focus({preventScroll:true});announce(t("saved"));
 }catch(e){announce(e.message,true);button.disabled=false;}
});
$("care-demo").onclick=async()=>{
 const button=$("care-demo");button.disabled=true;
 try{
  const health=await api("health");
  saveSession(await api("profiles","POST",{name:"Demo Resident",caregiver:"Demo Caregiver",email:health.mail_mode==="preview"?"care-demo@example.invalid":"",reminders_enabled:health.mail_mode==="preview",allergies:"Fictional example — no real patient information.",care_note:"This profile is a demonstration. Enter clinician-provided details for an actual care plan."}));
  const upcoming=localParts();
  await api("medications","POST",{name:"Prescribed medicine (demo)",dose:"Clinician-provided dose (demo)",instructions:"Fictional example for the care diary.",times:[upcoming.time],start_day:upcoming.day,active:true});
  const visit=localParts(new Date(Date.now()+45*60000));
  await api("appointments","POST",{title:"Care review (demo)",doctor:"Demo care professional",local_at:`${visit.day}T${visit.time}`,location:"Fictional example",notes:"No clinic booking is made.",status:"scheduled"});
  await api("reminders/check","POST");await load();announce(t("demoNotice"));$("profile-heading").tabIndex=-1;$("profile-heading").focus();
 }catch(e){announce(e.message,true);}finally{button.disabled=false;}
};
translations();
if(session)load().catch(e=>announce(e.message,true));
else api("health").catch(()=>announce(t("offline"),true));
