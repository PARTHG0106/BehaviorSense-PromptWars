/* Today's counts describe caregiver records, including doses scheduled for later today. */
export function summarizeDoses(doses) {
  const counts = {
    scheduled: doses.length,
    taken: 0,
    skipped: 0,
    pending: 0,
    overdue: 0,
  };
  for (const dose of doses) {
    if (Object.hasOwn(counts, dose.status) && dose.status !== "scheduled") {
      counts[dose.status]++;
    }
  }
  return { ...counts, confirmed: counts.taken + counts.skipped };
}
