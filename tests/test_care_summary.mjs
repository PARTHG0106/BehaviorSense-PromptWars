import test from "node:test";
import assert from "node:assert/strict";
import { summarizeDoses } from "../web/scripts/care-summary.js";

test("pending and overdue doses remain unconfirmed, including later doses", () => {
  const doses = Object.freeze([
    Object.freeze({ status: "taken" }),
    Object.freeze({ status: "skipped", note: "Caregiver reported a reason" }),
    Object.freeze({ status: "pending", due_at: "2099-01-01T00:00:00Z" }),
    Object.freeze({
      status: "overdue",
      note: "Video predicted a medication movement",
    }),
  ]);
  assert.deepEqual(summarizeDoses(doses), {
    scheduled: 4,
    taken: 1,
    skipped: 1,
    pending: 1,
    overdue: 1,
    confirmed: 2,
  });
});

test("empty plans report zero confirmations without inventing adherence", () => {
  assert.deepEqual(summarizeDoses([]), {
    scheduled: 0,
    taken: 0,
    skipped: 0,
    pending: 0,
    overdue: 0,
    confirmed: 0,
  });
});
