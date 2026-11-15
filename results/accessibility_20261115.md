# Accessibility and functional browser audit — 15 November 2026

The revised care interface and existing activity interface were tested in headless Chrome using Playwright and axe-core. Test records were fictional and email mode was preview. No real emails were sent.

| Audited view | Axe violations |
|---|---:|
| Care sign-in, desktop | 0 |
| Shared care plan, desktop | 0 |
| Care plan, mobile with extra-large text and high contrast | 0 |
| Care plan, Marathi on mobile | 0 |
| Activity page at 320 CSS pixels | 0 |
| Activity page, desktop | 0 |

The audit used WCAG 2 A/AA, 2.1 A/AA and 2.2 AA tags. Functional checks confirmed:

The reproducible test is included as `tests/test_accessibility.mjs` with pinned Node development dependencies and a dedicated GitHub Actions job.

- First keyboard focus reaches the skip link; Enter focuses the main care content.
- An explicit taken confirmation creates one immutable history row.
- Text and high-contrast preferences change the interface.
- Hindi and Marathi controls and document language update correctly.
- Viewports of 390 and 320 CSS pixels had matching content widths, with no horizontal page overflow.
- Browser JavaScript errors: **0**.

The screenshot in the README was taken from this fictional demo. The audit provides evidence for the exercised views and does not certify full WCAG conformance, clinical accuracy or untested assistive-technology combinations.
