import { Locale } from "./i18n";

export function isDraftTranslation(value: string): boolean {
  const trimmed = value.trim();
  return (
    trimmed.startsWith("[초안 번역]") ||
    trimmed.startsWith("[Korean draft pending]") ||
    trimmed === ""
  );
}

export function displayTranslationText(
  original: string,
  translated: string,
  locale: Locale,
  status?: "draft" | "ready" | "cached" | "fallback",
): string {
  const trimmed = translated.trim();
  if (!isDraftTranslation(trimmed)) return trimmed;
  return original;
}

export function displayPaperTitle(title: string, locale: Locale): string {
  if (title.trim() !== "Untitled paper") return title;
  return locale === "ko" ? "제목 없는 논문" : title;
}

export function displayAuthors(authors: string[], locale: Locale): string {
  if (authors.length === 1 && authors[0]?.trim() === "Unknown authors") {
    return locale === "ko" ? "저자 정보 없음" : "Unknown authors";
  }
  return authors.join(", ");
}

export function displayPaperSource(source: string, locale: Locale): string {
  if (source.trim() === "manual input") {
    return locale === "ko" ? "텍스트 붙여넣기" : "manual input";
  }
  return source;
}
