import type { Metadata } from "next";
import Nursery from "@/components/Nursery";
import { stageFor } from "@/lib/stages";

type Props = { searchParams: Promise<{ w?: string; l?: string }> };

/**
 * A shared link carries the model's first words in ?w= and its final loss in
 * ?l=, so the unfurl shows the sentence that particular model invented. Every
 * run produces a different one, which is the entire reason to share it.
 */
export async function generateMetadata({ searchParams }: Props): Promise<Metadata> {
  const { w, l } = await searchParams;
  const quote = (w ?? "").trim().slice(0, 140);

  if (!quote) {
    return {
      openGraph: { images: ["/api/og"] },
      twitter: { card: "summary_large_image", images: ["/api/og"] },
    };
  }

  const loss = Number(l);
  const stage = Number.isFinite(loss) ? stageFor(loss).label : null;
  const title = `"${quote}"`;
  const description = stage
    ? `A neural network raised in a browser tab. Final stage: ${stage}.`
    : "A neural network raised in a browser tab.";

  const image = `/api/og?w=${encodeURIComponent(quote)}${
    Number.isFinite(loss) ? `&l=${loss}` : ""
  }`;

  return {
    title,
    description,
    openGraph: { title, description, images: [image] },
    twitter: { card: "summary_large_image", title, description, images: [image] },
  };
}

export default function Page() {
  return <Nursery />;
}
