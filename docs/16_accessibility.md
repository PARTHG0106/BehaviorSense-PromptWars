# Accessibility in BehaviorSense Care

The care platform is designed for caregivers and older users who may need larger text, keyboard access or a different language. Its interface uses native HTML controls and visible text for statuses.

- A first-focus skip link moves keyboard focus to the main care content.
- Named landmarks, ordered headings, explicit field labels and help/error associations support screen-reader navigation.
- Native buttons, selects and disclosure controls support keyboard use. Focus is visible; error messages are announced and receive focus after a failed form submission.
- Controls have at least 44-pixel height, with large clickable labels for checkboxes.
- Text-size settings, high contrast and reduced motion persist as display preferences without patient data. System reduced-motion and forced-color preferences are respected.
- The complete care UI supports English, Hindi and Marathi and updates the document language. Patient-supplied names and prescription information are preserved.
- Pending, overdue, taken, skipped and delivery states contain readable labels. Their meaning does not rely on color.
- Cards and forms reflow on narrow screens; the history table provides labelled column headers and an accessible scroll region.
- Print styles provide a readable care-plan view.

## Verification

An axe-core audit using WCAG 2 A/AA, 2.1 A/AA and 2.2 AA tags reported zero violations in six views. Functional browser checks covered keyboard skip navigation, taken/skipped confirmations and notes, daily counts, medication creation, appointment attendance, printing, page navigation, language switching, text enlargement, high contrast, 390-pixel mobile view and 320-pixel reflow. No JavaScript errors were observed. See [the audit record](../results/accessibility_20261115.md).

Automated tests do not establish complete WCAG conformance. Live assistive-technology review with older users remains future work. Care information is entered by caregivers; UI accessibility and arithmetic evidence checks do not establish clinical accuracy.

The reproducible test is `tests/test_accessibility.mjs`. Install the Python serving requirements and run `npm ci`, then `npx playwright-core install --with-deps chromium` and `npm run test:accessibility`. Windows can use its installed Chrome through `BROWSER_EXECUTABLE`. The test starts a temporary preview-only care server, creates fictional records, checks all six views, and removes its private temporary database. A separate GitHub Actions job runs this audit.
