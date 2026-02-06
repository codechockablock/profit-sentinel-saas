"""
DORIAN ECONOMICS DOMAIN
=======================

Comprehensive economics knowledge for Dorian Core.

Includes:
1. Microeconomics
2. Macroeconomics
3. Finance
4. Markets and trade
5. Economic theories and schools
6. Key economists

Author: Joseph + Claude
Date: 2026-01-25
"""

ECON_CATEGORIES = [
    # =========================================================================
    # BRANCHES
    # =========================================================================
    ("economics", "field", "Study of resource allocation"),
    ("microeconomics", "economics", "Individual economic behavior"),
    ("macroeconomics", "economics", "Economy-wide phenomena"),
    ("behavioral_economics", "economics", "Psychology and economics"),
    ("econometrics", "economics", "Statistical economics"),
    ("international_economics", "economics", "Cross-border economics"),
    ("development_economics", "economics", "Economic development"),
    ("environmental_economics", "economics", "Environment and economy"),
    ("labor_economics", "economics", "Labor markets"),
    ("public_economics", "economics", "Government economics"),
    ("financial_economics", "economics", "Financial markets"),
    # =========================================================================
    # MICROECONOMICS
    # =========================================================================
    # Basic concepts
    ("scarcity", "concept", "Limited resources"),
    ("opportunity_cost", "concept", "Value of next best alternative"),
    ("marginal_utility", "concept", "Utility from one more unit"),
    ("diminishing_returns", "concept", "Decreasing marginal product"),
    ("comparative_advantage", "concept", "Lower opportunity cost"),
    ("absolute_advantage", "concept", "More efficient production"),
    # Supply and demand
    ("supply", "concept", "Quantity offered at price"),
    ("demand", "concept", "Quantity desired at price"),
    ("equilibrium", "concept", "Supply equals demand"),
    ("price_elasticity", "concept", "Responsiveness to price change"),
    ("elastic_demand", "price_elasticity", "Highly responsive to price"),
    ("inelastic_demand", "price_elasticity", "Unresponsive to price"),
    ("supply_curve", "concept", "Supply vs price relationship"),
    ("demand_curve", "concept", "Demand vs price relationship"),
    ("surplus", "concept", "Supply exceeds demand"),
    ("shortage", "concept", "Demand exceeds supply"),
    # Market structures
    ("market_structure", "concept", "Type of market organization"),
    ("perfect_competition", "market_structure", "Many firms, identical products"),
    ("monopoly", "market_structure", "Single seller"),
    ("oligopoly", "market_structure", "Few large sellers"),
    (
        "monopolistic_competition",
        "market_structure",
        "Many firms, differentiated products",
    ),
    ("monopsony", "market_structure", "Single buyer"),
    ("duopoly", "oligopoly", "Two sellers"),
    # Consumer theory
    ("utility", "concept", "Satisfaction from consumption"),
    ("indifference_curve", "concept", "Equal utility combinations"),
    ("budget_constraint", "concept", "Affordable combinations"),
    ("consumer_surplus", "concept", "Value minus price paid"),
    ("producer_surplus", "concept", "Price minus cost"),
    # Production theory
    ("production_function", "concept", "Input-output relationship"),
    ("marginal_product", "concept", "Output from one more input"),
    ("economies_of_scale", "concept", "Decreasing average cost"),
    ("diseconomies_of_scale", "concept", "Increasing average cost"),
    ("fixed_cost", "concept", "Cost independent of output"),
    ("variable_cost", "concept", "Cost dependent on output"),
    ("marginal_cost", "concept", "Cost of one more unit"),
    ("average_cost", "concept", "Total cost per unit"),
    # =========================================================================
    # MACROECONOMICS
    # =========================================================================
    # Key metrics
    ("gdp", "concept", "Gross Domestic Product"),
    ("nominal_gdp", "gdp", "GDP at current prices"),
    ("real_gdp", "gdp", "GDP adjusted for inflation"),
    ("gdp_per_capita", "gdp", "GDP per person"),
    ("gnp", "concept", "Gross National Product"),
    ("gni", "concept", "Gross National Income"),
    # Economic indicators
    ("economic_indicator", "concept", "Measure of economic health"),
    ("inflation", "economic_indicator", "Rising price level"),
    ("deflation", "economic_indicator", "Falling price level"),
    ("unemployment", "economic_indicator", "Joblessness rate"),
    ("employment_rate", "economic_indicator", "Employed percentage"),
    ("interest_rate", "economic_indicator", "Cost of borrowing"),
    ("exchange_rate", "economic_indicator", "Currency conversion rate"),
    ("trade_balance", "economic_indicator", "Exports minus imports"),
    ("current_account", "economic_indicator", "International transactions"),
    # Business cycles
    ("business_cycle", "concept", "Economic fluctuations"),
    ("expansion", "business_cycle", "Growing economy"),
    ("peak", "business_cycle", "Maximum output"),
    ("recession", "business_cycle", "Declining output"),
    ("trough", "business_cycle", "Minimum output"),
    ("depression", "recession", "Severe prolonged recession"),
    ("recovery", "business_cycle", "Return to growth"),
    # Unemployment types
    ("unemployment_type", "concept", "Category of unemployment"),
    ("frictional_unemployment", "unemployment_type", "Job search unemployment"),
    ("structural_unemployment", "unemployment_type", "Skills mismatch"),
    ("cyclical_unemployment", "unemployment_type", "Due to recession"),
    ("natural_unemployment", "unemployment_type", "Frictional plus structural"),
    # Inflation types
    ("inflation_type", "concept", "Category of inflation"),
    ("demand_pull", "inflation_type", "Excess demand causes"),
    ("cost_push", "inflation_type", "Rising costs cause"),
    ("hyperinflation", "inflation_type", "Extremely high inflation"),
    ("stagflation", "concept", "Inflation plus stagnation"),
    # =========================================================================
    # MONETARY ECONOMICS
    # =========================================================================
    ("money", "concept", "Medium of exchange"),
    ("currency", "money", "Physical money"),
    ("fiat_money", "money", "Government-declared money"),
    ("commodity_money", "money", "Valuable commodity as money"),
    ("money_supply", "concept", "Total money in economy"),
    ("m1", "money_supply", "Narrow money"),
    ("m2", "money_supply", "Broad money"),
    # Central banking
    ("central_bank", "institution", "Monetary authority"),
    ("federal_reserve", "central_bank", "US central bank"),
    ("ecb", "central_bank", "European Central Bank"),
    ("boe", "central_bank", "Bank of England"),
    ("boj", "central_bank", "Bank of Japan"),
    # Monetary policy
    ("monetary_policy", "policy", "Central bank actions"),
    ("open_market_operations", "monetary_policy", "Buying/selling securities"),
    ("reserve_requirement", "monetary_policy", "Required bank reserves"),
    ("discount_rate", "monetary_policy", "Rate for bank borrowing"),
    ("quantitative_easing", "monetary_policy", "Large-scale asset purchases"),
    ("interest_rate_policy", "monetary_policy", "Setting interest rates"),
    # =========================================================================
    # FISCAL POLICY
    # =========================================================================
    ("fiscal_policy", "policy", "Government spending and taxation"),
    ("government_spending", "fiscal_policy", "Public expenditure"),
    ("taxation", "fiscal_policy", "Government revenue collection"),
    ("budget_deficit", "concept", "Spending exceeds revenue"),
    ("budget_surplus", "concept", "Revenue exceeds spending"),
    ("national_debt", "concept", "Accumulated deficits"),
    ("austerity", "fiscal_policy", "Reduced spending"),
    ("stimulus", "fiscal_policy", "Increased spending"),
    # Tax types
    ("tax_type", "taxation", "Category of tax"),
    ("income_tax", "tax_type", "Tax on earnings"),
    ("corporate_tax", "tax_type", "Tax on business profits"),
    ("sales_tax", "tax_type", "Tax on purchases"),
    ("vat", "sales_tax", "Value-added tax"),
    ("capital_gains_tax", "tax_type", "Tax on investment gains"),
    ("property_tax", "tax_type", "Tax on real estate"),
    ("tariff", "tax_type", "Tax on imports"),
    ("progressive_tax", "tax_type", "Higher rate for higher income"),
    ("regressive_tax", "tax_type", "Higher rate for lower income"),
    ("flat_tax", "tax_type", "Same rate for all"),
    # =========================================================================
    # FINANCIAL MARKETS
    # =========================================================================
    ("financial_market", "market", "Market for financial instruments"),
    ("stock_market", "financial_market", "Equity trading"),
    ("bond_market", "financial_market", "Debt trading"),
    ("forex_market", "financial_market", "Currency trading"),
    ("derivatives_market", "financial_market", "Derivatives trading"),
    ("commodities_market", "financial_market", "Commodities trading"),
    ("money_market", "financial_market", "Short-term debt"),
    ("capital_market", "financial_market", "Long-term securities"),
    # Financial instruments
    ("financial_instrument", "concept", "Tradeable financial asset"),
    ("stock", "financial_instrument", "Equity ownership"),
    ("common_stock", "stock", "Voting shares"),
    ("preferred_stock", "stock", "Priority dividends"),
    ("bond", "financial_instrument", "Debt security"),
    ("government_bond", "bond", "Government debt"),
    ("corporate_bond", "bond", "Company debt"),
    ("treasury_bill", "government_bond", "Short-term government debt"),
    ("municipal_bond", "bond", "Local government debt"),
    # Derivatives
    ("derivative", "financial_instrument", "Value derived from underlying"),
    ("option", "derivative", "Right to buy/sell"),
    ("call_option", "option", "Right to buy"),
    ("put_option", "option", "Right to sell"),
    ("future", "derivative", "Obligation to buy/sell"),
    ("forward", "derivative", "Customized future"),
    ("swap", "derivative", "Exchange of cash flows"),
    ("credit_default_swap", "swap", "Insurance on default"),
    # Investment concepts
    ("portfolio", "concept", "Collection of investments"),
    ("diversification", "concept", "Spreading risk"),
    ("return", "concept", "Investment gain"),
    ("risk", "concept", "Uncertainty of return"),
    ("risk_premium", "concept", "Extra return for risk"),
    ("alpha", "concept", "Excess return"),
    ("beta", "concept", "Market sensitivity"),
    ("sharpe_ratio", "concept", "Risk-adjusted return"),
    ("volatility", "concept", "Price variation"),
    ("liquidity", "concept", "Ease of trading"),
    # =========================================================================
    # BANKING AND FINANCE
    # =========================================================================
    ("bank", "institution", "Financial intermediary"),
    ("commercial_bank", "bank", "Retail banking"),
    ("investment_bank", "bank", "Securities and advisory"),
    ("credit_union", "bank", "Member-owned bank"),
    ("hedge_fund", "institution", "Alternative investment fund"),
    ("mutual_fund", "institution", "Pooled investment"),
    ("etf", "mutual_fund", "Exchange-traded fund"),
    ("index_fund", "mutual_fund", "Tracks market index"),
    ("pension_fund", "institution", "Retirement savings"),
    ("insurance_company", "institution", "Risk transfer"),
    # Banking concepts
    ("deposit", "concept", "Money placed in bank"),
    ("loan", "concept", "Borrowed money"),
    ("mortgage", "loan", "Property-secured loan"),
    ("credit", "concept", "Borrowed purchasing power"),
    ("interest", "concept", "Cost of borrowing"),
    ("compound_interest", "interest", "Interest on interest"),
    ("principal", "concept", "Original loan amount"),
    ("collateral", "concept", "Loan security"),
    ("default", "concept", "Failure to repay"),
    ("bankruptcy", "concept", "Legal insolvency"),
    # =========================================================================
    # INTERNATIONAL ECONOMICS
    # =========================================================================
    ("trade", "concept", "Exchange of goods/services"),
    ("export", "trade", "Selling abroad"),
    ("import", "trade", "Buying from abroad"),
    ("free_trade", "trade", "Unrestricted trade"),
    ("protectionism", "trade", "Trade restrictions"),
    ("trade_agreement", "concept", "Trade treaty"),
    ("trade_war", "concept", "Retaliatory tariffs"),
    ("trade_deficit", "concept", "Imports exceed exports"),
    ("trade_surplus", "concept", "Exports exceed imports"),
    # International institutions
    ("international_institution", "institution", "Global economic body"),
    ("imf", "international_institution", "International Monetary Fund"),
    ("world_bank", "international_institution", "Development bank"),
    ("wto", "international_institution", "World Trade Organization"),
    ("oecd", "international_institution", "Developed nations group"),
    ("g7", "international_institution", "Seven major economies"),
    ("g20", "international_institution", "Twenty major economies"),
    # Exchange rates
    ("exchange_rate_system", "concept", "Currency valuation system"),
    ("floating_exchange", "exchange_rate_system", "Market-determined rate"),
    ("fixed_exchange", "exchange_rate_system", "Pegged rate"),
    ("currency_peg", "fixed_exchange", "Fixed to another currency"),
    ("devaluation", "concept", "Lowering currency value"),
    ("appreciation", "concept", "Rising currency value"),
    ("depreciation", "concept", "Falling currency value"),
    # =========================================================================
    # ECONOMIC THEORIES AND SCHOOLS
    # =========================================================================
    ("economic_school", "theory", "School of economic thought"),
    ("classical_economics", "economic_school", "Smith, Ricardo, Say"),
    ("neoclassical_economics", "economic_school", "Marginalist revolution"),
    ("keynesian_economics", "economic_school", "Demand management"),
    ("monetarism", "economic_school", "Money supply focus"),
    ("austrian_economics", "economic_school", "Individual action"),
    ("chicago_school", "economic_school", "Free market focus"),
    ("supply_side", "economic_school", "Tax cuts for growth"),
    ("post_keynesian", "economic_school", "Extended Keynesianism"),
    ("mmt", "economic_school", "Modern Monetary Theory"),
    ("institutional_economics", "economic_school", "Role of institutions"),
    # Key theories
    ("economic_theory", "theory", "Economic model/theory"),
    ("invisible_hand", "economic_theory", "Market self-regulation"),
    ("says_law", "economic_theory", "Supply creates demand"),
    ("quantity_theory", "economic_theory", "Money and prices"),
    ("phillips_curve", "economic_theory", "Inflation-unemployment tradeoff"),
    ("efficient_market_hypothesis", "economic_theory", "Markets reflect info"),
    ("rational_expectations", "economic_theory", "Forward-looking agents"),
    ("game_theory", "economic_theory", "Strategic interaction"),
    ("nash_equilibrium", "game_theory", "No profitable deviation"),
    # =========================================================================
    # ECONOMISTS
    # =========================================================================
    ("economist", "person", "Economics expert"),
    ("adam_smith", "economist", "Father of economics"),
    ("david_ricardo", "economist", "Comparative advantage"),
    ("karl_marx", "economist", "Capital, class struggle"),
    ("john_maynard_keynes", "economist", "Keynesian economics"),
    ("milton_friedman", "economist", "Monetarism"),
    ("friedrich_hayek", "economist", "Austrian economics"),
    ("paul_samuelson", "economist", "Mathematical economics"),
    ("john_nash", "economist", "Game theory"),
    ("joseph_stiglitz", "economist", "Information economics"),
    ("paul_krugman", "economist", "Trade theory"),
    ("ben_bernanke", "economist", "Fed chairman, 2008 crisis"),
    ("janet_yellen", "economist", "Fed chair, Treasury"),
    # =========================================================================
    # ECONOMIC PHENOMENA
    # =========================================================================
    ("economic_phenomenon", "concept", "Economic event/pattern"),
    ("bubble", "economic_phenomenon", "Asset price inflation"),
    ("crash", "economic_phenomenon", "Sudden price collapse"),
    ("bank_run", "economic_phenomenon", "Mass deposit withdrawal"),
    ("contagion", "economic_phenomenon", "Crisis spreading"),
    ("moral_hazard", "economic_phenomenon", "Risk-taking due to insurance"),
    ("adverse_selection", "economic_phenomenon", "Information asymmetry"),
    ("market_failure", "economic_phenomenon", "Inefficient outcome"),
    ("externality", "market_failure", "Third-party effects"),
    ("public_good", "market_failure", "Non-excludable, non-rival"),
    ("free_rider", "market_failure", "Benefiting without paying"),
    # Historical events
    ("great_depression", "economic_phenomenon", "1930s depression"),
    ("great_recession", "economic_phenomenon", "2008 financial crisis"),
    ("dot_com_bubble", "bubble", "Late 1990s tech bubble"),
    ("housing_bubble", "bubble", "2000s housing bubble"),
]

ECON_FACTS = [
    # Fundamentals
    ("economics", "studies", "resource_allocation"),
    ("scarcity", "fundamental_to", "economics"),
    ("opportunity_cost", "is_a", "fundamental_concept"),
    # Supply and demand
    ("supply", "increases_with", "price"),
    ("demand", "decreases_with", "price"),
    ("equilibrium", "where", "supply_equals_demand"),
    # Market structures
    ("perfect_competition", "has", "many_sellers"),
    ("monopoly", "has", "one_seller"),
    ("oligopoly", "has", "few_sellers"),
    # GDP
    ("gdp", "measures", "economic_output"),
    ("gdp", "equals", "c_plus_i_plus_g_plus_nx"),
    ("real_gdp", "adjusted_for", "inflation"),
    # Inflation
    ("inflation", "is", "rising_prices"),
    ("deflation", "is", "falling_prices"),
    ("hyperinflation", "example", "zimbabwe_venezuela"),
    # Unemployment
    ("unemployment", "natural_rate", "4_to_5_percent"),
    ("frictional_unemployment", "is", "temporary"),
    ("structural_unemployment", "is", "skills_mismatch"),
    # Monetary policy
    ("central_bank", "controls", "money_supply"),
    ("federal_reserve", "is", "us_central_bank"),
    ("interest_rate_increase", "decreases", "inflation"),
    ("interest_rate_decrease", "stimulates", "economy"),
    ("quantitative_easing", "increases", "money_supply"),
    # Fiscal policy
    ("government_spending", "increases", "gdp"),
    ("taxation", "decreases", "disposable_income"),
    ("deficit_spending", "increases", "national_debt"),
    # Schools of thought
    ("keynesian_economics", "emphasizes", "demand_management"),
    ("monetarism", "emphasizes", "money_supply"),
    ("classical_economics", "emphasizes", "free_markets"),
    ("adam_smith", "wrote", "wealth_of_nations"),
    ("keynes", "wrote", "general_theory"),
    ("friedman", "advocated", "monetarism"),
    # Key relationships
    ("phillips_curve", "shows", "inflation_unemployment_tradeoff"),
    ("invisible_hand", "proposed_by", "adam_smith"),
    ("comparative_advantage", "proposed_by", "david_ricardo"),
    # Finance
    ("stock", "represents", "ownership"),
    ("bond", "represents", "debt"),
    ("diversification", "reduces", "risk"),
    ("higher_risk", "requires", "higher_return"),
    # International
    ("free_trade", "increases", "efficiency"),
    ("tariffs", "are", "protectionist"),
    ("exchange_rate", "determines", "currency_value"),
    # Crises
    ("great_depression", "started", "1929"),
    ("great_recession", "started", "2008"),
    ("2008_crisis", "caused_by", "housing_bubble"),
]


def load_economics_into_core(core, agent_id: str = None) -> int:
    if agent_id is None:
        econ_agent = core.register_agent(
            "economics_loader", domain="economics", can_verify=True
        )
        agent_id = econ_agent.agent_id

    count = 0

    print(f"  Loading {len(ECON_CATEGORIES)} economics categories...")
    for name, parent, description in ECON_CATEGORIES:
        if core.ontology:
            from dorian_ontology import Category

            parent_cat = core.ontology.categories.get(parent)
            parent_level = parent_cat.level if parent_cat else 3
            core.ontology._add_category(
                Category(
                    name=name,
                    description=description,
                    parent=parent,
                    domain="economics",
                    level=parent_level + 1,
                )
            )
        result = core.write(
            name,
            "subtype_of",
            parent,
            agent_id,
            source="economics_ontology",
            check_contradictions=False,
        )
        if result.success:
            count += 1

    print(f"  Loading {len(ECON_FACTS)} economics facts...")
    for s, p, o in ECON_FACTS:
        result = core.write(
            s, p, o, agent_id, source="economics_knowledge", check_contradictions=False
        )
        if result.success:
            count += 1

    print(f"  Total: {count} economics facts loaded")
    return count


if __name__ == "__main__":
    print(f"Economics: {len(ECON_CATEGORIES)} categories, {len(ECON_FACTS)} facts")
