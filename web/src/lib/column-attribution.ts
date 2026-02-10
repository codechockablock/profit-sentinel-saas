/**
 * Column Attribution — Data lineage for analysis results
 *
 * Builds attribution metadata that tells users exactly which CSV column
 * each data point came from, how derived fields were calculated, and
 * what each metric is used to detect.
 *
 * Zero backend changes — everything is computed from the mapping state
 * that already persists in the AnalysisDashboard component.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ColumnAttribution {
  /** Internal field key: "quantity", "cost", "margin", etc. */
  normalizedField: string
  /** Short label shown in results: "QOH", "Cost", "Margin" */
  displayLabel: string
  /** Full human-readable name for tooltip header */
  displayLabelFull: string

  // Source lineage (populated at runtime from confirmedMapping)
  /** Original CSV column name(s): ["In Stock Qty."] */
  sourceColumns: string[]
  /** Excel-style column letter(s): ["G"] */
  sourceColumnLetters: string[]

  // Derivation metadata (from static registry)
  /** True if calculated from other mapped columns */
  isDerived: boolean
  /** True if no source column was found in the user's data */
  isDefaulted: boolean
  /** Formula expression: "(revenue - cost) / revenue × 100" */
  formula: string | null
  /** Pretty formula using actual source column names */
  formulaDisplay: string | null
  /** Internal fields this derived field depends on */
  dependsOn: string[] | null
  /** Column names we searched for (if defaulted) */
  searchedColumns: string[] | null
  /** Value we used when no source column found */
  defaultValue: string | null

  // Education
  /** What this field means and how it's used */
  explanation: string
  /** Leak types this field helps detect */
  detects: string[]
  /** Industry benchmark, if applicable */
  benchmark: string | null
  /** Source of the benchmark data */
  benchmarkSource: string | null
}

/** Static definition before runtime mapping is applied */
interface AttributionDefinition {
  displayLabel: string
  displayLabelFull: string
  isDerived: boolean
  formula: string | null
  dependsOn: string[] | null
  searchedColumns: string[] | null
  defaultValue: string | null
  explanation: string
  detects: string[]
  benchmark: string | null
  benchmarkSource: string | null
}

// ---------------------------------------------------------------------------
// Static Registry
// ---------------------------------------------------------------------------

const ATTRIBUTION_REGISTRY: Record<string, AttributionDefinition> = {
  sku: {
    displayLabel: 'SKU',
    displayLabelFull: 'Stock Keeping Unit',
    isDerived: false,
    formula: null,
    dependsOn: null,
    searchedColumns: null,
    defaultValue: null,
    explanation: 'Unique product identifier from your POS system. Every item in your inventory has one.',
    detects: [],
    benchmark: null,
    benchmarkSource: null,
  },
  description: {
    displayLabel: 'Desc.',
    displayLabelFull: 'Product Description',
    isDerived: false,
    formula: null,
    dependsOn: null,
    searchedColumns: null,
    defaultValue: null,
    explanation: 'Product name or description from your POS. Helps you identify items at a glance.',
    detects: [],
    benchmark: null,
    benchmarkSource: null,
  },
  quantity: {
    displayLabel: 'QOH',
    displayLabelFull: 'Quantity on Hand',
    isDerived: false,
    formula: null,
    dependsOn: null,
    searchedColumns: null,
    defaultValue: '0',
    explanation: 'Your current inventory count from your POS. We use this to detect stock-level anomalies.',
    detects: [
      'Negative Inventory',
      'Dead Stock',
      'Low Stock',
      'Overstock',
      'Severe Inventory Deficit',
    ],
    benchmark: null,
    benchmarkSource: null,
  },
  cost: {
    displayLabel: 'Cost',
    displayLabelFull: 'Unit Cost',
    isDerived: false,
    formula: null,
    dependsOn: null,
    searchedColumns: null,
    defaultValue: null,
    explanation: 'What you paid for this item from your vendor. Critical for margin calculations.',
    detects: [
      'Margin Erosion',
      'Negative Profit',
      'Shrinkage Pattern',
      'Zero Cost Anomaly',
    ],
    benchmark: null,
    benchmarkSource: null,
  },
  revenue: {
    displayLabel: 'Retail',
    displayLabelFull: 'Retail Price',
    isDerived: false,
    formula: null,
    dependsOn: null,
    searchedColumns: null,
    defaultValue: null,
    explanation: 'Current selling price in your POS. Compared against cost to find pricing issues.',
    detects: [
      'Margin Leak',
      'Price Discrepancy',
      'Negative Profit',
    ],
    benchmark: null,
    benchmarkSource: null,
  },
  sold: {
    displayLabel: 'Sold',
    displayLabelFull: 'Units Sold',
    isDerived: false,
    formula: null,
    dependsOn: null,
    searchedColumns: null,
    defaultValue: '0',
    explanation: 'Number of units sold in the reporting period. Drives velocity-based detections.',
    detects: [
      'Dead Stock',
      'Low Stock',
      'Severe Inventory Deficit',
    ],
    benchmark: null,
    benchmarkSource: null,
  },
  margin: {
    displayLabel: 'Margin',
    displayLabelFull: 'Profit Margin %',
    isDerived: true,
    formula: '(revenue − cost) ÷ revenue × 100',
    dependsOn: ['cost', 'revenue'],
    searchedColumns: null,
    defaultValue: null,
    explanation: 'Gross margin percentage. If your CSV includes a margin column, we use it directly. Otherwise we calculate it from cost and retail price.',
    detects: [
      'Margin Erosion',
      'Margin Leak',
      'Shrinkage Pattern',
    ],
    benchmark: '35–45%',
    benchmarkSource: 'Hardware retail industry average (NHPA / Do It Best)',
  },
  sub_total: {
    displayLabel: 'Value',
    displayLabelFull: 'Inventory Value',
    isDerived: true,
    formula: 'quantity × cost',
    dependsOn: ['quantity', 'cost'],
    searchedColumns: null,
    defaultValue: null,
    explanation: 'Total value of on-hand stock for this item. Helps prioritize which dead stock or overstock items tie up the most capital.',
    detects: [
      'Overstock',
      'Dead Stock',
    ],
    benchmark: null,
    benchmarkSource: null,
  },
  vendor: {
    displayLabel: 'Vendor',
    displayLabelFull: 'Vendor / Supplier',
    isDerived: false,
    formula: null,
    dependsOn: null,
    searchedColumns: null,
    defaultValue: null,
    explanation: 'The vendor or supplier for this product. Used to group issues by vendor for call prep.',
    detects: [
      'Vendor Short-Ship',
    ],
    benchmark: null,
    benchmarkSource: null,
  },
  category: {
    displayLabel: 'Category',
    displayLabelFull: 'Product Category',
    isDerived: false,
    formula: null,
    dependsOn: null,
    searchedColumns: null,
    defaultValue: null,
    explanation: 'Product category or department. Helps identify category-wide trends vs isolated issues.',
    detects: [
      'Category Mix',
    ],
    benchmark: null,
    benchmarkSource: null,
  },

  // --- Rust pipeline fields (shown when M1 adapter is active) ---
  store_id: {
    displayLabel: 'Store',
    displayLabelFull: 'Store ID',
    isDerived: false,
    formula: null,
    dependsOn: null,
    searchedColumns: ['Store', 'Location', 'Branch', 'Site', 'Warehouse'],
    defaultValue: 'default',
    explanation: 'Identifies which location this inventory belongs to. For single-store analysis, we default this.',
    detects: [],
    benchmark: null,
    benchmarkSource: null,
  },
  days_since_receipt: {
    displayLabel: 'Days Since Rcpt',
    displayLabelFull: 'Days Since Last Receipt',
    isDerived: true,
    formula: 'today − last purchase date',
    dependsOn: ['last_purchase_date'],
    searchedColumns: ['Last Pur.', 'Last Purchase', 'Last Received', 'Receipt Date'],
    defaultValue: '30',
    explanation: 'How long since you last received this item from a vendor. Items not received in 90+ days may indicate stale inventory.',
    detects: [
      'Dead Stock',
    ],
    benchmark: '90+ days = dead stock risk',
    benchmarkSource: 'Industry standard',
  },
  is_damaged: {
    displayLabel: 'Damaged',
    displayLabelFull: 'Damaged Flag',
    isDerived: false,
    formula: null,
    dependsOn: null,
    searchedColumns: ['Damaged', 'Defective', 'Damaged Flag', 'Status'],
    defaultValue: 'false',
    explanation: 'Whether this item is marked as damaged or defective in your POS.',
    detects: [
      'Shrinkage Pattern',
    ],
    benchmark: null,
    benchmarkSource: null,
  },
  on_order_qty: {
    displayLabel: 'On Order',
    displayLabelFull: 'On Order Quantity',
    isDerived: false,
    formula: null,
    dependsOn: null,
    searchedColumns: ['On Order', 'PO Qty', 'Ordered', 'Backordered'],
    defaultValue: '0',
    explanation: 'Units currently on order from vendors. Helps detect short-ship patterns.',
    detects: [
      'Vendor Short-Ship',
    ],
    benchmark: null,
    benchmarkSource: null,
  },
  is_seasonal: {
    displayLabel: 'Seasonal',
    displayLabelFull: 'Seasonal Item',
    isDerived: false,
    formula: null,
    dependsOn: null,
    searchedColumns: ['Seasonal', 'Season', 'Seasonal Flag'],
    defaultValue: 'false',
    explanation: 'Whether this item is seasonal. Seasonal dead stock may not be a problem if it\'s off-season.',
    detects: [
      'Dead Stock (seasonal)',
    ],
    benchmark: null,
    benchmarkSource: null,
  },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Convert a 0-based column index to an Excel-style letter.
 * 0 → "A", 25 → "Z", 26 → "AA", 27 → "AB", etc.
 */
export function columnIndexToLetter(index: number): string {
  let result = ''
  let n = index
  while (n >= 0) {
    result = String.fromCharCode((n % 26) + 65) + result
    n = Math.floor(n / 26) - 1
  }
  return result
}

/**
 * Invert a mapping from {sourceCol: targetField} to {targetField: sourceCol[]}.
 * Skips empty/null target values. Groups multiple sources mapping to the same target.
 */
export function invertMapping(
  mapping: Record<string, string>
): Record<string, string[]> {
  const inverted: Record<string, string[]> = {}
  for (const [source, target] of Object.entries(mapping)) {
    if (!target || !target.trim()) continue
    if (!inverted[target]) {
      inverted[target] = []
    }
    inverted[target].push(source)
  }
  return inverted
}

// ---------------------------------------------------------------------------
// Build Attributions
// ---------------------------------------------------------------------------

/**
 * Build attribution metadata for all known fields, using the confirmed
 * column mapping and the original CSV column list.
 *
 * This is designed to be called once in a useMemo — it's pure and
 * deterministic given the same inputs.
 */
export function buildAttributions(
  confirmedMapping: Record<string, string>,
  originalColumns: string[]
): Record<string, ColumnAttribution> {
  const inverted = invertMapping(confirmedMapping)
  const result: Record<string, ColumnAttribution> = {}

  for (const [field, def] of Object.entries(ATTRIBUTION_REGISTRY)) {
    // Look up which source column(s) map to this field
    const sourceColumns = inverted[field] ?? []
    const hasDirectSource = sourceColumns.length > 0

    // Compute column letters from positions in original CSV
    const sourceColumnLetters = sourceColumns.map((col) => {
      const idx = originalColumns.indexOf(col)
      return idx >= 0 ? columnIndexToLetter(idx) : '?'
    })

    // Determine if this field is defaulted (no source + not derivable)
    let isDefaulted = false
    if (!hasDirectSource) {
      if (def.isDerived && def.dependsOn) {
        // Derived field — check if its dependencies have sources
        const allDepsResolved = def.dependsOn.every(
          (dep) => (inverted[dep] ?? []).length > 0
        )
        isDefaulted = !allDepsResolved
      } else {
        // Non-derived field — defaulted if no source column
        isDefaulted = def.defaultValue !== null
      }
    }

    // Build the dynamic formula display using actual source column names
    let formulaDisplay: string | null = null
    if (def.isDerived && def.formula && def.dependsOn) {
      formulaDisplay = def.formula
      // Replace generic field names with actual source column names
      for (const dep of def.dependsOn) {
        const depSources = inverted[dep] ?? []
        if (depSources.length > 0) {
          // Find the registry entry for the dependency to get its display label
          const depDef = ATTRIBUTION_REGISTRY[dep]
          const depLabel = depDef?.displayLabel ?? dep
          // Replace the dep name in the formula with the quoted source column name
          formulaDisplay = formulaDisplay.replace(
            new RegExp(`\\b${depLabel.toLowerCase()}\\b`, 'i'),
            `"${depSources[0]}"`
          )
        }
      }
    }

    // Build the human-readable explanation
    let explanation = ''
    if (hasDirectSource) {
      const colDescs = sourceColumns
        .map((col, i) => `"${col}" (Column ${sourceColumnLetters[i]})`)
        .join(' and ')
      explanation = `Mapped from your ${colDescs}. ${def.explanation}`
    } else if (isDefaulted) {
      const searched = def.searchedColumns ?? []
      const searchedStr =
        searched.length > 0
          ? ` We looked for: ${searched.map((s) => `"${s}"`).join(', ')}.`
          : ''
      explanation = `Not found in your data.${searchedStr} Defaulted to: ${def.defaultValue ?? 'N/A'}. ${def.explanation}`
    } else if (def.isDerived) {
      explanation = `Calculated from your data. ${def.explanation}`
    } else {
      explanation = def.explanation
    }

    result[field] = {
      normalizedField: field,
      displayLabel: def.displayLabel,
      displayLabelFull: def.displayLabelFull,
      sourceColumns,
      sourceColumnLetters,
      isDerived: def.isDerived,
      isDefaulted,
      formula: def.formula,
      formulaDisplay,
      dependsOn: def.dependsOn,
      searchedColumns: def.searchedColumns,
      defaultValue: def.defaultValue,
      explanation,
      detects: def.detects,
      benchmark: def.benchmark,
      benchmarkSource: def.benchmarkSource,
    }
  }

  return result
}
