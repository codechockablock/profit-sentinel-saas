import {
  columnIndexToLetter,
  invertMapping,
  buildAttributions,
  type ColumnAttribution,
} from '../column-attribution'

// ---------------------------------------------------------------------------
// columnIndexToLetter
// ---------------------------------------------------------------------------

describe('columnIndexToLetter', () => {
  it('converts single-letter indices', () => {
    expect(columnIndexToLetter(0)).toBe('A')
    expect(columnIndexToLetter(1)).toBe('B')
    expect(columnIndexToLetter(5)).toBe('F')
    expect(columnIndexToLetter(25)).toBe('Z')
  })

  it('converts double-letter indices', () => {
    expect(columnIndexToLetter(26)).toBe('AA')
    expect(columnIndexToLetter(27)).toBe('AB')
    expect(columnIndexToLetter(51)).toBe('AZ')
    expect(columnIndexToLetter(52)).toBe('BA')
  })

  it('handles large indices', () => {
    // 702 = AAA (26 + 26*26 + 0)
    expect(columnIndexToLetter(702)).toBe('AAA')
  })
})

// ---------------------------------------------------------------------------
// invertMapping
// ---------------------------------------------------------------------------

describe('invertMapping', () => {
  it('inverts a standard mapping', () => {
    const mapping = {
      'In Stock Qty.': 'quantity',
      'Cost': 'cost',
      'Retail': 'revenue',
    }
    const inverted = invertMapping(mapping)
    expect(inverted).toEqual({
      quantity: ['In Stock Qty.'],
      cost: ['Cost'],
      revenue: ['Retail'],
    })
  })

  it('skips empty target values', () => {
    const mapping = {
      'SKU': 'sku',
      'Vendor SKU': '',
      'Description Short': '',
    }
    const inverted = invertMapping(mapping)
    expect(inverted).toEqual({
      sku: ['SKU'],
    })
  })

  it('groups multiple sources mapping to the same target', () => {
    const mapping = {
      'Sug. Retail': 'revenue',
      'Retail': 'revenue',
    }
    const inverted = invertMapping(mapping)
    expect(inverted.revenue).toHaveLength(2)
    expect(inverted.revenue).toContain('Sug. Retail')
    expect(inverted.revenue).toContain('Retail')
  })

  it('handles empty mapping', () => {
    expect(invertMapping({})).toEqual({})
  })
})

// ---------------------------------------------------------------------------
// buildAttributions — Paladin POS CSV
// ---------------------------------------------------------------------------

describe('buildAttributions with Paladin POS columns', () => {
  // Real Paladin POS column list from production CSV
  const originalColumns = [
    'SKU', 'Vendor SKU', 'Description ', 'Description Short',
    'Pkg. Qty.', 'Inventoried Qty.', 'In Stock Qty.', 'Qty. Difference',
    'Cost', 'Sub Total', 'Changed', 'Sug. Retail', 'Retail',
    'Profit Margin %', 'Allow Retail Edit', 'Last Perpetual Inventory Date',
    'Real Perpetual Inventory Date', 'Barcode', 'Vendor', 'Category',
    'Dpt.', 'BIN', 'Status', 'Sold', 'Last Sale', 'Pur.', 'Last Pur.',
    'Sku Was Added', 'Formula',
  ]

  const confirmedMapping: Record<string, string> = {
    'SKU': 'sku',
    'Description ': 'description',
    'In Stock Qty.': 'quantity',
    'Cost': 'cost',
    'Retail': 'revenue',
    'Sold': 'sold',
    'Profit Margin %': 'margin',
    'Sub Total': 'sub_total',
    'Vendor': 'vendor',
    'Category': 'category',
    // These are intentionally skipped (empty):
    'Vendor SKU': '',
    'Description Short': '',
    'Pkg. Qty.': '',
    'Dpt.': '',
  }

  let attributions: Record<string, ColumnAttribution>

  beforeAll(() => {
    attributions = buildAttributions(confirmedMapping, originalColumns)
  })

  it('maps direct fields correctly', () => {
    const qty = attributions.quantity
    expect(qty).toBeDefined()
    expect(qty.sourceColumns).toEqual(['In Stock Qty.'])
    expect(qty.sourceColumnLetters).toEqual(['G'])
    expect(qty.isDerived).toBe(false)
    expect(qty.isDefaulted).toBe(false)
    expect(qty.displayLabel).toBe('QOH')
    expect(qty.displayLabelFull).toBe('Quantity on Hand')
  })

  it('maps cost correctly with column letter', () => {
    const cost = attributions.cost
    expect(cost.sourceColumns).toEqual(['Cost'])
    expect(cost.sourceColumnLetters).toEqual(['I'])
    expect(cost.isDerived).toBe(false)
    expect(cost.isDefaulted).toBe(false)
  })

  it('maps revenue correctly', () => {
    const rev = attributions.revenue
    expect(rev.sourceColumns).toEqual(['Retail'])
    expect(rev.sourceColumnLetters).toEqual(['M'])
    expect(rev.displayLabel).toBe('Retail')
  })

  it('marks margin as derived but not defaulted (since it has a direct source)', () => {
    // margin is marked isDerived in the registry, but the user mapped
    // "Profit Margin %" directly to it, so it also has a sourceColumn
    const margin = attributions.margin
    expect(margin.isDerived).toBe(true)
    expect(margin.isDefaulted).toBe(false)
    expect(margin.sourceColumns).toEqual(['Profit Margin %'])
    expect(margin.sourceColumnLetters).toEqual(['N'])
    expect(margin.benchmark).toBe('35–45%')
    expect(margin.benchmarkSource).toContain('NHPA')
  })

  it('marks sub_total as derived with source column', () => {
    const sub = attributions.sub_total
    expect(sub.isDerived).toBe(true)
    expect(sub.isDefaulted).toBe(false)
    expect(sub.sourceColumns).toEqual(['Sub Total'])
  })

  it('marks store_id as defaulted', () => {
    const store = attributions.store_id
    expect(store.isDefaulted).toBe(true)
    expect(store.sourceColumns).toEqual([])
    expect(store.defaultValue).toBe('default')
    expect(store.searchedColumns).toContain('Store')
    expect(store.searchedColumns).toContain('Location')
    expect(store.explanation).toContain('Not found in your data')
    expect(store.explanation).toContain('Defaulted to: default')
  })

  it('marks is_damaged as defaulted', () => {
    const dmg = attributions.is_damaged
    expect(dmg.isDefaulted).toBe(true)
    expect(dmg.defaultValue).toBe('false')
    expect(dmg.searchedColumns).toContain('Damaged')
  })

  it('includes detects for each field', () => {
    expect(attributions.quantity.detects).toContain('Negative Inventory')
    expect(attributions.quantity.detects).toContain('Dead Stock')
    expect(attributions.cost.detects).toContain('Margin Erosion')
    expect(attributions.revenue.detects).toContain('Price Discrepancy')
    expect(attributions.sold.detects).toContain('Low Stock')
  })

  it('builds explanation with source column name and letter', () => {
    const qty = attributions.quantity
    expect(qty.explanation).toContain('"In Stock Qty."')
    expect(qty.explanation).toContain('Column G')
  })

  it('handles all known fields', () => {
    // Should have entries for all registry fields
    expect(Object.keys(attributions).length).toBeGreaterThanOrEqual(10)
    expect(attributions.sku).toBeDefined()
    expect(attributions.description).toBeDefined()
    expect(attributions.quantity).toBeDefined()
    expect(attributions.cost).toBeDefined()
    expect(attributions.revenue).toBeDefined()
    expect(attributions.sold).toBeDefined()
    expect(attributions.margin).toBeDefined()
    expect(attributions.vendor).toBeDefined()
    expect(attributions.category).toBeDefined()
  })
})

// ---------------------------------------------------------------------------
// buildAttributions — empty mapping (all defaulted)
// ---------------------------------------------------------------------------

describe('buildAttributions with empty mapping', () => {
  it('marks non-derived fields with defaults as defaulted', () => {
    const attributions = buildAttributions({}, [])
    // quantity has defaultValue: '0', so it's defaulted
    expect(attributions.quantity.isDefaulted).toBe(true)
    expect(attributions.quantity.sourceColumns).toEqual([])
  })

  it('marks derived fields as defaulted when deps are missing', () => {
    const attributions = buildAttributions({}, [])
    // margin depends on cost + revenue, neither mapped
    expect(attributions.margin.isDefaulted).toBe(true)
  })

  it('does not crash on empty inputs', () => {
    expect(() => buildAttributions({}, [])).not.toThrow()
    const attrs = buildAttributions({}, [])
    expect(Object.keys(attrs).length).toBeGreaterThan(0)
  })
})

// ---------------------------------------------------------------------------
// buildAttributions — partial mapping
// ---------------------------------------------------------------------------

describe('buildAttributions with partial mapping', () => {
  it('resolves derived margin when cost and revenue are mapped', () => {
    const mapping = {
      'Unit Cost': 'cost',
      'Sell Price': 'revenue',
    }
    const columns = ['Unit Cost', 'Sell Price']
    const attrs = buildAttributions(mapping, columns)

    // margin is derived and both deps are resolved
    expect(attrs.margin.isDerived).toBe(true)
    expect(attrs.margin.isDefaulted).toBe(false)
    expect(attrs.margin.formula).toContain('revenue')
  })

  it('marks margin as defaulted when only cost is mapped', () => {
    const mapping = {
      'Unit Cost': 'cost',
    }
    const columns = ['Unit Cost']
    const attrs = buildAttributions(mapping, columns)

    // margin depends on cost + revenue, but revenue is missing
    expect(attrs.margin.isDerived).toBe(true)
    expect(attrs.margin.isDefaulted).toBe(true)
  })
})
