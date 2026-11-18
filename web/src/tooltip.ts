// Formula-tip HTML (vanilla util/tooltip.js `formulaHTML`), shown in the
// shared popover from Tip.tsx.
import { escapeHtml, formatGiB } from "./format";

export type FormulaSubterm = {
  label?: string;
  symbol?: string;
  symbolic?: string;
  substituted?: string;
  result?: number | string;
  result_bytes?: number;
  note?: string;
  subterms?: FormulaSubterm[];
  defs?: FormulaDef[];
};

/** An intermediate quantity a formula names, at the shown point: its law
 *  with the point's numbers bound (`phase_formulas.FormulaDefs`). */
/** A named intermediate quantity: its values line (`formula`) and value,
 *  and — for a name with a declared definition (`s_pairs`) — that
 *  definition in names, printed first. */
export type FormulaDef = { name: string; formula: string; value: string; definition?: string };

export type FormulaFields = {
  title?: string;
  /** Capacity-formula alias of `title`. */
  label?: string;
  symbolic?: string;
  substituted?: string;
  result?: string | number;
  result_bytes?: number;
  note?: string;
  /** Capacity-formula alias of `note`. */
  description?: string;
  per_layer_result?: string;
  cnt?: number;
  cnt_unit?: string;
  copy_factors?: { n: number; unit: string }[];
  subterms?: FormulaSubterm[];
  /** The named intermediate quantities, listed under the values line. */
  defs?: FormulaDef[];
};

function resultLine(f: { result?: string | number; result_bytes?: number }): string | undefined {
  if (f.result != null && f.result !== "") return String(f.result);
  if (f.result_bytes != null) return formatGiB(f.result_bytes);
  return undefined;
}

/** Parent formula line: the node's own identity when it has one. Children's
 *  IDs fill in only when the parent has none; they still expand below. */
function formulaLine({ symbolic, subterms }: FormulaFields): string {
  if (symbolic) return symbolic;
  const ids = (Array.isArray(subterms) ? subterms : [])
    .map((st) => st.label || st.symbol)
    .filter(Boolean);
  return ids.join(" + ");
}

/** A declared definition whose definition or values line is longer than
 *  this, in characters, shows only `name = value`, both lines folded under
 *  it (`<details>`): the dense phase's fit on a line, a top-k's or a pooled
 *  union's closed form does not. */
const DEFINITION_FOLD_CHARS = 80;

export function formulaHTML({ title, label, symbolic, substituted, result, result_bytes, note, description, per_layer_result, cnt, cnt_unit, copy_factors, subterms, defs }: FormulaFields) {
  const out: string[] = [];
  const heading = title || label;
  if (heading) out.push(`<div class="text-10 text-gray-400 uppercase tracking-wide mb-1">${escapeHtml(heading)}</div>`);
  const rowHTML = (rowLabel: string, bodyHTML: string, cls: string) =>
    `<div class="flex gap-3 items-start"><span class="text-gray-400 w-20 shrink-0">${escapeHtml(rowLabel)}</span><span class="${cls} break-words min-w-0">${bodyHTML}</span></div>`;
  const row = (rowLabel: string, body: string, cls: string) => rowHTML(rowLabel, escapeHtml(body), cls);
  const formula = formulaLine({ symbolic, subterms });
  if (formula) out.push(row("formula", formula, "text-indigo-700 italic"));
  if (substituted) out.push(row("values", substituted, "text-gray-700"));
  (Array.isArray(defs) ? defs : []).forEach((d, i) => {
    const label = i === 0 ? "where" : "";
    if (d.definition && (d.definition.length > DEFINITION_FOLD_CHARS
                         || d.formula.length > DEFINITION_FOLD_CHARS)) {
      // A long definition or values line (a top-k's or a pooled union's
      // closed form) shows only its value; both lines fold under it and the
      // reader opens them.
      const valueLine = escapeHtml(`${d.name} = ${d.value}`);
      const folded = `<details class="mt-0.5"><summary class="cursor-pointer select-none text-gray-400">definition</summary>`
        + `<div class="text-indigo-700 italic">${escapeHtml(`${d.name} = ${d.definition}`)}</div>`
        + `<div class="text-gray-600 mt-0.5">${escapeHtml(`= ${d.formula}`)}</div></details>`;
      out.push(rowHTML(label, valueLine + folded, "text-gray-600"));
      return;
    }
    const parts = d.definition ? [d.name, d.definition, d.formula, d.value] : [d.name, d.formula, d.value];
    out.push(row(label, parts.join(" = "), "text-gray-600"));
  });
  const factors = Array.isArray(copy_factors) ? copy_factors.filter((f) => f && f.n > 1) : [];
  if (per_layer_result && factors.length) {
    const onlyLayers = factors.length === 1 && factors[0].unit === "layers";
    out.push(row(onlyLayers ? "per layer" : "per copy", per_layer_result, "text-gray-700"));
    out.push(row("×", factors.map((f) => `${f.n} ${f.unit}`).join(" × "), "text-gray-500 italic"));
  } else if (per_layer_result && cnt && cnt > 1) {
    out.push(row("per layer", per_layer_result, "text-gray-700"));
    out.push(row("×", `${cnt} ${cnt_unit || "layers"}`, "text-gray-500 italic"));
  }
  const shown = resultLine({ result, result_bytes });
  if (shown) out.push(row("=", shown, "text-amber-700 font-semibold"));
  const footnote = note || description;
  if (footnote) out.push(`<div class="text-10 text-gray-400 italic mt-1">${escapeHtml(footnote)}</div>`);
  if (Array.isArray(subterms) && subterms.length) {
    for (const st of subterms) {
      out.push(`<div class="mt-2 pt-2 border-t border-gray-200 pl-3 border-l-2 border-l-gray-100">${formulaHTML({
        title: st.label || st.symbol,
        symbolic: st.symbolic,
        substituted: st.substituted,
        result: st.result,
        result_bytes: st.result_bytes,
        note: st.note,
        subterms: st.subterms,
        defs: st.defs,
      })}</div>`);
    }
  }
  return out.join("");
}
