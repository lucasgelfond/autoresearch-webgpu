/**
 * Tiny i18n entry point: pick the active locale and re-export its strings.
 * Adding a locale = add `src/lib/i18n/<code>.ts` with the same shape (see
 * `Messages` in ./fr.ts) and register it in `locales` below.
 */
import { fr, type Messages } from './fr';

export { type Messages };

export const locales = { fr } as const;

export type Locale = keyof typeof locales;

/** The app ships in French only, for now. */
export const locale: Locale = 'fr';

/** Strings for the active locale. */
export const t: Messages = locales[locale];

/** Value for `<html lang>` and `hreflang`. */
export const htmlLang: string = locale;
