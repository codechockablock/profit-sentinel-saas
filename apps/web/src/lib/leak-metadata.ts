/**
 * Leak Detection Metadata - User-friendly explanations, impact estimates, and recommendations
 *
 * This module provides all the context needed to make raw leak scores
 * understandable and actionable for store owners.
 */

export interface LeakMetadata {
  key: string;
  title: string;
  description: string;
  plainEnglish: string;
  impactFormula: string;
  recommendations: string[];
  severity: 'critical' | 'high' | 'medium' | 'low';
  color: string;
  bgColor: string;
  borderColor: string;
  icon: string;
  tooltipExamples: string[];
}

export const LEAK_METADATA: Record<string, LeakMetadata> = {
  high_margin_leak: {
    key: 'high_margin_leak',
    title: 'High Margin Leak',
    description: 'Items selling below expected profit margin',
    plainEnglish: 'You\'re leaving money on the table! These items are selling for less profit than they should. This often happens when costs go up but prices don\'t, or items are accidentally discounted too much.',
    impactFormula: 'Lost margin × Units sold',
    recommendations: [
      'Review and update retail prices for flagged items',
      'Check if vendor costs have increased recently',
      'Verify no unauthorized discounts are being applied',
      'Consider bundling low-margin items with high-margin ones'
    ],
    severity: 'critical',
    color: '#ef4444',
    bgColor: 'rgba(239, 68, 68, 0.1)',
    borderColor: 'rgba(239, 68, 68, 0.3)',
    icon: 'trending-down',
    tooltipExamples: [
      'Item costs $8, sells for $9 (12.5% margin vs 30% target)',
      'Vendor raised prices but POS wasn\'t updated'
    ]
  },

  negative_inventory: {
    key: 'negative_inventory',
    title: 'Negative Inventory',
    description: 'Items showing impossible negative quantities',
    plainEnglish: 'Your system says you have LESS than zero of these items - which is impossible! This usually means sales weren\'t recorded properly, items were stolen, or receiving was skipped. This is a red flag for shrink or data errors.',
    impactFormula: 'Negative units × Item cost',
    recommendations: [
      'Immediately investigate each negative SKU',
      'Check if receiving was skipped for recent shipments',
      'Review POS transaction logs for voids/returns',
      'Consider physical count to verify actual stock',
      'Train staff on proper receiving procedures'
    ],
    severity: 'critical',
    color: '#dc2626',
    bgColor: 'rgba(220, 38, 38, 0.1)',
    borderColor: 'rgba(220, 38, 38, 0.3)',
    icon: 'alert-circle',
    tooltipExamples: [
      'System shows -5 units (sold 5 that weren\'t received)',
      'Possible theft or receiving paperwork missed'
    ]
  },

  low_stock: {
    key: 'low_stock',
    title: 'Low Stock Alert',
    description: 'Items at risk of stockout',
    plainEnglish: 'These items are running dangerously low! You\'re about to lose sales because customers will find empty shelves. Fast-moving items need reordering NOW.',
    impactFormula: 'Est. lost sales × Profit margin',
    recommendations: [
      'Place emergency orders for critical items',
      'Set up automatic reorder points in your POS',
      'Review lead times with vendors',
      'Consider safety stock for top sellers'
    ],
    severity: 'high',
    color: '#f59e0b',
    bgColor: 'rgba(245, 158, 11, 0.1)',
    borderColor: 'rgba(245, 158, 11, 0.3)',
    icon: 'alert-triangle',
    tooltipExamples: [
      'Only 2 left, sells 10/week - will stockout in 2 days',
      'Popular item with no backup stock'
    ]
  },

  shrinkage_pattern: {
    key: 'shrinkage_pattern',
    title: 'Shrinkage Pattern',
    description: 'Inventory discrepancies indicating loss',
    plainEnglish: 'There\'s a gap between what your system says you have and what\'s actually there. This "shrinkage" could be theft, damage, vendor fraud, or clerical errors. It\'s money walking out the door.',
    impactFormula: 'Missing units × Item cost',
    recommendations: [
      'Conduct cycle counts on flagged items',
      'Review security footage for high-value items',
      'Check vendor deliveries match invoices',
      'Implement better receiving verification',
      'Consider locked display for theft-prone items'
    ],
    severity: 'high',
    color: '#f97316',
    bgColor: 'rgba(249, 115, 22, 0.1)',
    borderColor: 'rgba(249, 115, 22, 0.3)',
    icon: 'shield-alert',
    tooltipExamples: [
      'Expected 50, found 42 - 8 units missing ($200 loss)',
      'Pattern suggests systematic theft or vendor short-shipping'
    ]
  },

  margin_erosion: {
    key: 'margin_erosion',
    title: 'Margin Erosion',
    description: 'Items with declining profitability over time',
    plainEnglish: 'These items used to be profitable but margins are shrinking. Costs may be creeping up, or you\'re discounting more than you realize. Death by a thousand cuts!',
    impactFormula: 'Margin decline × Revenue',
    recommendations: [
      'Compare current vs historical margins',
      'Negotiate better vendor pricing',
      'Review discount patterns and promotions',
      'Consider discontinuing chronically low-margin items'
    ],
    severity: 'high',
    color: '#ec4899',
    bgColor: 'rgba(236, 72, 153, 0.1)',
    borderColor: 'rgba(236, 72, 153, 0.3)',
    icon: 'trending-down',
    tooltipExamples: [
      'Was 35% margin last year, now only 18%',
      'Vendor raised costs 3x but price unchanged'
    ]
  },

  dead_item: {
    key: 'dead_item',
    title: 'Dead Inventory',
    description: 'Items with no sales in 90+ days',
    plainEnglish: 'This inventory is just sitting there collecting dust and tying up your cash. Every dollar stuck in dead stock is a dollar you can\'t use for fast-moving items or bills.',
    impactFormula: 'Units on hand × Unit cost',
    recommendations: [
      'Run clearance sale on dead items',
      'Return to vendor if possible',
      'Bundle with popular items',
      'Donate for tax write-off',
      'Don\'t reorder - let it sell through'
    ],
    severity: 'medium',
    color: '#6b7280',
    bgColor: 'rgba(107, 114, 128, 0.1)',
    borderColor: 'rgba(107, 114, 128, 0.3)',
    icon: 'package-x',
    tooltipExamples: [
      '25 units sitting for 6 months = $500 tied up',
      'Seasonal item that missed its window'
    ]
  },

  overstock: {
    key: 'overstock',
    title: 'Overstock',
    description: 'Excess inventory above optimal levels',
    plainEnglish: 'You\'ve got way more of these items than you can sell in a reasonable time. That\'s cash sitting on shelves instead of in your bank account earning interest or paying bills.',
    impactFormula: 'Excess units × Cost × Carrying rate',
    recommendations: [
      'Slow down or pause reorders',
      'Run promotion to move excess',
      'Negotiate return to vendor',
      'Transfer to other locations if applicable'
    ],
    severity: 'medium',
    color: '#3b82f6',
    bgColor: 'rgba(59, 130, 246, 0.1)',
    borderColor: 'rgba(59, 130, 246, 0.3)',
    icon: 'boxes',
    tooltipExamples: [
      '200 units, sells 5/month = 40 months of stock!',
      'Over-ordered for promotion that didn\'t happen'
    ]
  },

  price_discrepancy: {
    key: 'price_discrepancy',
    title: 'Price Discrepancy',
    description: 'Items with pricing inconsistencies',
    plainEnglish: 'The selling price doesn\'t match what it should be. Could be a pricing error, an outdated sale price that wasn\'t removed, or items scanning at the wrong price.',
    impactFormula: 'Price difference × Units sold',
    recommendations: [
      'Audit POS price file against vendor suggested retail',
      'Check for expired promotional pricing',
      'Verify shelf tags match system prices',
      'Review price override patterns'
    ],
    severity: 'low',
    color: '#8b5cf6',
    bgColor: 'rgba(139, 92, 246, 0.1)',
    borderColor: 'rgba(139, 92, 246, 0.3)',
    icon: 'tag',
    tooltipExamples: [
      'MSRP is $29.99 but selling at $24.99',
      'Sale ended but price wasn\'t updated back'
    ]
  }
};

/**
 * Get severity badge styling
 */
export function getSeverityBadge(severity: string): { label: string; className: string } {
  const badges: Record<string, { label: string; className: string }> = {
    critical: {
      label: 'CRITICAL',
      className: 'bg-red-500/20 text-red-400 border-red-500/30'
    },
    high: {
      label: 'HIGH',
      className: 'bg-orange-500/20 text-orange-400 border-orange-500/30'
    },
    medium: {
      label: 'MEDIUM',
      className: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
    },
    low: {
      label: 'LOW',
      className: 'bg-blue-500/20 text-blue-400 border-blue-500/30'
    }
  };
  return badges[severity] || badges.low;
}

/**
 * Convert raw score (0-1) to user-friendly risk label
 */
export function scoreToRiskLabel(score: number): { label: string; percentage: string; className: string } {
  const percentage = Math.round(score * 100);

  if (score >= 0.8) {
    return {
      label: 'Critical Risk',
      percentage: `${percentage}%`,
      className: 'text-red-400 bg-red-500/20 border-red-500/30'
    };
  } else if (score >= 0.6) {
    return {
      label: 'High Risk',
      percentage: `${percentage}%`,
      className: 'text-orange-400 bg-orange-500/20 border-orange-500/30'
    };
  } else if (score >= 0.4) {
    return {
      label: 'Medium Risk',
      percentage: `${percentage}%`,
      className: 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30'
    };
  } else if (score >= 0.2) {
    return {
      label: 'Low Risk',
      percentage: `${percentage}%`,
      className: 'text-blue-400 bg-blue-500/20 border-blue-500/30'
    };
  } else {
    return {
      label: 'Minimal',
      percentage: `${percentage}%`,
      className: 'text-green-400 bg-green-500/20 border-green-500/30'
    };
  }
}

/**
 * Format dollar amount for display
 */
export function formatDollarImpact(amount: number): string {
  if (amount >= 1000000) {
    return `$${(amount / 1000000).toFixed(1)}M`;
  } else if (amount >= 1000) {
    return `$${(amount / 1000).toFixed(1)}K`;
  } else {
    return `$${amount.toFixed(0)}`;
  }
}

/**
 * Calculate estimated impact for a single item based on leak type
 */
export function estimateItemImpact(
  leakType: string,
  score: number,
  baseValue: number = 100
): { low: number; high: number } {
  // Base estimation - will be replaced with real data from backend
  const multipliers: Record<string, number> = {
    high_margin_leak: 50,
    negative_inventory: 75,
    low_stock: 30,
    shrinkage_pattern: 60,
    margin_erosion: 40,
    dead_item: 25,
    overstock: 15,
    price_discrepancy: 20
  };

  const multiplier = multipliers[leakType] || 25;
  const base = score * multiplier * baseValue / 100;

  return {
    low: Math.round(base * 0.7),
    high: Math.round(base * 1.3)
  };
}
