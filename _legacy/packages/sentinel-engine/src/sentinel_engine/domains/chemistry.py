"""
DORIAN CHEMISTRY DOMAIN
=======================

Comprehensive chemistry knowledge for Dorian Core.

Includes:
1. Elements and periodic table
2. Chemical bonding and structures
3. Reactions and mechanisms
4. Organic chemistry
5. Inorganic chemistry
6. Physical chemistry
7. Biochemistry basics

Author: Joseph + Claude
Date: 2026-01-25
"""

from typing import Dict, List, Set, Tuple

# =============================================================================
# CHEMISTRY CATEGORIES
# =============================================================================

CHEM_CATEGORIES = [
    # =========================================================================
    # FUNDAMENTAL CONCEPTS
    # =========================================================================
    (
        "chemical_substance",
        "physical_object",
        "A form of matter with constant composition",
    ),
    ("element", "chemical_substance", "A substance made of one type of atom"),
    ("compound", "chemical_substance", "A substance made of multiple elements"),
    (
        "mixture",
        "chemical_substance",
        "A combination of substances not chemically bonded",
    ),
    ("homogeneous_mixture", "mixture", "A uniform mixture"),
    ("heterogeneous_mixture", "mixture", "A non-uniform mixture"),
    ("solution", "homogeneous_mixture", "A homogeneous mixture of solute and solvent"),
    ("colloid", "mixture", "A mixture with dispersed particles"),
    ("suspension", "heterogeneous_mixture", "A mixture with suspended particles"),
    # Atomic structure
    ("subatomic_particle", "particle", "A particle smaller than an atom"),
    ("atomic_orbital", "quantum_state", "Region of electron probability"),
    ("s_orbital", "atomic_orbital", "Spherical orbital"),
    ("p_orbital", "atomic_orbital", "Dumbbell-shaped orbital"),
    ("d_orbital", "atomic_orbital", "Cloverleaf orbital"),
    ("f_orbital", "atomic_orbital", "Complex orbital"),
    ("electron_shell", "structure", "Energy level containing electrons"),
    ("valence_shell", "electron_shell", "Outermost electron shell"),
    ("valence_electron", "electron", "Electron in valence shell"),
    # =========================================================================
    # ELEMENTS - BY CATEGORY
    # =========================================================================
    ("element_category", "abstract", "Category of chemical elements"),
    # Main categories
    ("alkali_metal", "element_category", "Group 1 elements"),
    ("alkaline_earth_metal", "element_category", "Group 2 elements"),
    ("transition_metal", "element_category", "d-block elements"),
    ("post_transition_metal", "element_category", "Metals after transition metals"),
    ("metalloid", "element_category", "Elements with metal/nonmetal properties"),
    ("nonmetal", "element_category", "Non-metallic elements"),
    ("halogen", "element_category", "Group 17 elements"),
    ("noble_gas", "element_category", "Group 18 elements"),
    ("lanthanide", "element_category", "Lanthanide series"),
    ("actinide", "element_category", "Actinide series"),
    # Specific elements - Period 1-3
    ("hydrogen", "element", "Element 1 - H"),
    ("helium", "element", "Element 2 - He"),
    ("lithium", "element", "Element 3 - Li"),
    ("beryllium", "element", "Element 4 - Be"),
    ("boron", "element", "Element 5 - B"),
    ("carbon", "element", "Element 6 - C"),
    ("nitrogen", "element", "Element 7 - N"),
    ("oxygen", "element", "Element 8 - O"),
    ("fluorine", "element", "Element 9 - F"),
    ("neon", "element", "Element 10 - Ne"),
    ("sodium", "element", "Element 11 - Na"),
    ("magnesium", "element", "Element 12 - Mg"),
    ("aluminum", "element", "Element 13 - Al"),
    ("silicon", "element", "Element 14 - Si"),
    ("phosphorus", "element", "Element 15 - P"),
    ("sulfur", "element", "Element 16 - S"),
    ("chlorine", "element", "Element 17 - Cl"),
    ("argon", "element", "Element 18 - Ar"),
    # Period 4
    ("potassium", "element", "Element 19 - K"),
    ("calcium", "element", "Element 20 - Ca"),
    ("scandium", "element", "Element 21 - Sc"),
    ("titanium", "element", "Element 22 - Ti"),
    ("vanadium", "element", "Element 23 - V"),
    ("chromium", "element", "Element 24 - Cr"),
    ("manganese", "element", "Element 25 - Mn"),
    ("iron", "element", "Element 26 - Fe"),
    ("cobalt", "element", "Element 27 - Co"),
    ("nickel", "element", "Element 28 - Ni"),
    ("copper", "element", "Element 29 - Cu"),
    ("zinc", "element", "Element 30 - Zn"),
    ("gallium", "element", "Element 31 - Ga"),
    ("germanium", "element", "Element 32 - Ge"),
    ("arsenic", "element", "Element 33 - As"),
    ("selenium", "element", "Element 34 - Se"),
    ("bromine", "element", "Element 35 - Br"),
    ("krypton", "element", "Element 36 - Kr"),
    # Period 5-6 key elements
    ("silver", "element", "Element 47 - Ag"),
    ("tin", "element", "Element 50 - Sn"),
    ("iodine", "element", "Element 53 - I"),
    ("xenon", "element", "Element 54 - Xe"),
    ("barium", "element", "Element 56 - Ba"),
    ("gold", "element", "Element 79 - Au"),
    ("mercury", "element", "Element 80 - Hg"),
    ("lead", "element", "Element 82 - Pb"),
    ("platinum", "element", "Element 78 - Pt"),
    ("tungsten", "element", "Element 74 - W"),
    ("uranium", "element", "Element 92 - U"),
    # =========================================================================
    # CHEMICAL BONDING
    # =========================================================================
    ("chemical_bond", "relation", "Attraction between atoms"),
    ("intramolecular_bond", "chemical_bond", "Bond within a molecule"),
    ("intermolecular_force", "chemical_bond", "Force between molecules"),
    # Intramolecular
    ("covalent_bond", "intramolecular_bond", "Shared electron bond"),
    ("single_bond", "covalent_bond", "One shared pair"),
    ("double_bond", "covalent_bond", "Two shared pairs"),
    ("triple_bond", "covalent_bond", "Three shared pairs"),
    ("polar_covalent", "covalent_bond", "Unequal electron sharing"),
    ("nonpolar_covalent", "covalent_bond", "Equal electron sharing"),
    ("coordinate_bond", "covalent_bond", "Both electrons from one atom"),
    ("ionic_bond", "intramolecular_bond", "Electrostatic attraction"),
    ("metallic_bond", "intramolecular_bond", "Delocalized electrons in metals"),
    # Intermolecular
    ("hydrogen_bond", "intermolecular_force", "H bonded to N, O, or F"),
    ("van_der_waals", "intermolecular_force", "Weak intermolecular force"),
    ("london_dispersion", "van_der_waals", "Temporary dipole force"),
    ("dipole_dipole", "intermolecular_force", "Permanent dipole attraction"),
    ("ion_dipole", "intermolecular_force", "Ion-polar molecule interaction"),
    # =========================================================================
    # MOLECULAR STRUCTURE
    # =========================================================================
    ("molecular_geometry", "structure", "3D arrangement of atoms"),
    ("linear_geometry", "molecular_geometry", "180° bond angle"),
    ("trigonal_planar", "molecular_geometry", "120° bond angle"),
    ("tetrahedral", "molecular_geometry", "109.5° bond angle"),
    ("trigonal_bipyramidal", "molecular_geometry", "90° and 120° angles"),
    ("octahedral", "molecular_geometry", "90° bond angles"),
    ("bent_geometry", "molecular_geometry", "Angular shape"),
    ("trigonal_pyramidal", "molecular_geometry", "Pyramid shape"),
    ("molecular_orbital", "quantum_state", "Orbital in a molecule"),
    ("bonding_orbital", "molecular_orbital", "Lower energy, bonding"),
    ("antibonding_orbital", "molecular_orbital", "Higher energy, antibonding"),
    ("sigma_bond", "covalent_bond", "Head-on orbital overlap"),
    ("pi_bond", "covalent_bond", "Side-by-side orbital overlap"),
    ("hybridization", "abstract", "Mixing of atomic orbitals"),
    ("sp_hybrid", "hybridization", "Linear hybridization"),
    ("sp2_hybrid", "hybridization", "Trigonal planar hybridization"),
    ("sp3_hybrid", "hybridization", "Tetrahedral hybridization"),
    ("sp3d_hybrid", "hybridization", "Trigonal bipyramidal hybridization"),
    ("sp3d2_hybrid", "hybridization", "Octahedral hybridization"),
    # =========================================================================
    # CHEMICAL REACTIONS
    # =========================================================================
    ("chemical_reaction", "process", "Transformation of substances"),
    # Reaction types
    ("synthesis_reaction", "chemical_reaction", "A + B → AB"),
    ("decomposition_reaction", "chemical_reaction", "AB → A + B"),
    ("single_replacement", "chemical_reaction", "A + BC → AC + B"),
    ("double_replacement", "chemical_reaction", "AB + CD → AD + CB"),
    ("combustion", "chemical_reaction", "Reaction with oxygen producing heat"),
    ("oxidation_reduction", "chemical_reaction", "Electron transfer reaction"),
    ("redox", "oxidation_reduction", "Oxidation-reduction reaction"),
    ("oxidation", "redox", "Loss of electrons"),
    ("reduction", "redox", "Gain of electrons"),
    ("acid_base_reaction", "chemical_reaction", "Proton transfer reaction"),
    ("neutralization", "acid_base_reaction", "Acid + base → salt + water"),
    ("precipitation", "chemical_reaction", "Formation of insoluble solid"),
    ("hydrolysis", "chemical_reaction", "Reaction with water"),
    ("polymerization", "chemical_reaction", "Joining monomers"),
    ("addition_polymerization", "polymerization", "Monomers add directly"),
    (
        "condensation_polymerization",
        "polymerization",
        "Monomers join losing small molecule",
    ),
    # Reaction characteristics
    ("exothermic", "property", "Releases heat"),
    ("endothermic", "property", "Absorbs heat"),
    ("reversible_reaction", "chemical_reaction", "Can proceed both directions"),
    ("irreversible_reaction", "chemical_reaction", "Proceeds in one direction"),
    # =========================================================================
    # ACIDS AND BASES
    # =========================================================================
    ("acid", "chemical_substance", "Proton donor"),
    ("base", "chemical_substance", "Proton acceptor"),
    ("strong_acid", "acid", "Completely dissociates"),
    ("weak_acid", "acid", "Partially dissociates"),
    ("strong_base", "base", "Completely dissociates"),
    ("weak_base", "base", "Partially dissociates"),
    # Specific acids
    ("hydrochloric_acid", "strong_acid", "HCl"),
    ("sulfuric_acid", "strong_acid", "H2SO4"),
    ("nitric_acid", "strong_acid", "HNO3"),
    ("phosphoric_acid", "weak_acid", "H3PO4"),
    ("acetic_acid", "weak_acid", "CH3COOH"),
    ("carbonic_acid", "weak_acid", "H2CO3"),
    # Specific bases
    ("sodium_hydroxide", "strong_base", "NaOH"),
    ("potassium_hydroxide", "strong_base", "KOH"),
    ("ammonia", "weak_base", "NH3"),
    ("sodium_bicarbonate", "weak_base", "NaHCO3"),
    # pH
    ("ph_scale", "measurement", "Measure of acidity"),
    ("acidic", "property", "pH < 7"),
    ("neutral", "property", "pH = 7"),
    ("basic", "property", "pH > 7"),
    ("alkaline", "basic", "pH > 7"),
    ("buffer", "solution", "Resists pH change"),
    # =========================================================================
    # ORGANIC CHEMISTRY
    # =========================================================================
    ("organic_compound", "compound", "Carbon-based compound"),
    # Hydrocarbons
    ("hydrocarbon", "organic_compound", "Contains only C and H"),
    ("alkane", "hydrocarbon", "Single bonds only - CnH2n+2"),
    ("alkene", "hydrocarbon", "Contains C=C double bond"),
    ("alkyne", "hydrocarbon", "Contains C≡C triple bond"),
    ("aromatic", "hydrocarbon", "Contains benzene ring"),
    ("cycloalkane", "hydrocarbon", "Ring of single bonds"),
    # Specific hydrocarbons
    ("methane", "alkane", "CH4 - simplest alkane"),
    ("ethane", "alkane", "C2H6"),
    ("propane", "alkane", "C3H8"),
    ("butane", "alkane", "C4H10"),
    ("ethene", "alkene", "C2H4 - ethylene"),
    ("propene", "alkene", "C3H6"),
    ("ethyne", "alkyne", "C2H2 - acetylene"),
    ("benzene", "aromatic", "C6H6 - aromatic ring"),
    ("toluene", "aromatic", "Methylbenzene"),
    ("naphthalene", "aromatic", "Fused benzene rings"),
    # Functional groups
    ("functional_group", "structure", "Reactive group of atoms"),
    ("hydroxyl_group", "functional_group", "-OH group"),
    ("carbonyl_group", "functional_group", "C=O group"),
    ("carboxyl_group", "functional_group", "-COOH group"),
    ("amino_group", "functional_group", "-NH2 group"),
    ("aldehyde_group", "functional_group", "-CHO group"),
    ("ketone_group", "functional_group", "C=O between carbons"),
    ("ester_group", "functional_group", "-COO- group"),
    ("ether_group", "functional_group", "C-O-C group"),
    ("amide_group", "functional_group", "-CONH2 group"),
    ("nitro_group", "functional_group", "-NO2 group"),
    ("sulfhydryl_group", "functional_group", "-SH group"),
    ("phosphate_group", "functional_group", "-PO4 group"),
    # Compound classes
    ("alcohol", "organic_compound", "Contains hydroxyl group"),
    ("aldehyde", "organic_compound", "Contains aldehyde group"),
    ("ketone", "organic_compound", "Contains ketone group"),
    ("carboxylic_acid", "organic_compound", "Contains carboxyl group"),
    ("ester", "organic_compound", "Contains ester group"),
    ("ether", "organic_compound", "Contains ether group"),
    ("amine", "organic_compound", "Contains amino group"),
    ("amide", "organic_compound", "Contains amide group"),
    ("nitrile", "organic_compound", "Contains -CN group"),
    # Specific organic compounds
    ("methanol", "alcohol", "CH3OH - wood alcohol"),
    ("ethanol", "alcohol", "C2H5OH - drinking alcohol"),
    ("propanol", "alcohol", "C3H7OH"),
    ("glycerol", "alcohol", "C3H8O3 - three OH groups"),
    ("formaldehyde", "aldehyde", "HCHO"),
    ("acetaldehyde", "aldehyde", "CH3CHO"),
    ("acetone", "ketone", "CH3COCH3"),
    ("formic_acid", "carboxylic_acid", "HCOOH"),
    ("ethyl_acetate", "ester", "CH3COOC2H5"),
    ("diethyl_ether", "ether", "C2H5OC2H5"),
    # Isomerism
    ("isomer", "compound", "Same formula, different structure"),
    ("structural_isomer", "isomer", "Different connectivity"),
    ("stereoisomer", "isomer", "Same connectivity, different 3D"),
    ("geometric_isomer", "stereoisomer", "Cis-trans isomerism"),
    ("optical_isomer", "stereoisomer", "Mirror image isomers"),
    ("enantiomer", "optical_isomer", "Non-superimposable mirror images"),
    ("diastereomer", "stereoisomer", "Non-mirror image stereoisomers"),
    ("chiral", "property", "Has non-superimposable mirror image"),
    ("achiral", "property", "Superimposable with mirror image"),
    # =========================================================================
    # INORGANIC CHEMISTRY
    # =========================================================================
    ("inorganic_compound", "compound", "Non-carbon-based compound"),
    # Oxides
    ("oxide", "inorganic_compound", "Compound with oxygen"),
    ("metal_oxide", "oxide", "Metal combined with oxygen"),
    ("nonmetal_oxide", "oxide", "Nonmetal combined with oxygen"),
    ("water", "oxide", "H2O - hydrogen oxide"),
    ("carbon_dioxide", "oxide", "CO2"),
    ("carbon_monoxide", "oxide", "CO"),
    ("sulfur_dioxide", "oxide", "SO2"),
    ("nitrogen_dioxide", "oxide", "NO2"),
    ("iron_oxide", "metal_oxide", "Fe2O3 - rust"),
    ("aluminum_oxide", "metal_oxide", "Al2O3"),
    # Salts
    ("salt", "inorganic_compound", "Ionic compound from acid-base"),
    ("sodium_chloride", "salt", "NaCl - table salt"),
    ("potassium_chloride", "salt", "KCl"),
    ("calcium_carbonate", "salt", "CaCite - limestone"),
    ("sodium_sulfate", "salt", "Na2SO4"),
    ("ammonium_nitrate", "salt", "NH4NO3"),
    # Coordination compounds
    ("coordination_compound", "inorganic_compound", "Metal with ligands"),
    ("ligand", "chemical_substance", "Molecule bonded to metal"),
    ("complex_ion", "ion", "Metal ion with ligands"),
    ("chelate", "coordination_compound", "Multi-dentate ligand complex"),
    # =========================================================================
    # PHYSICAL CHEMISTRY
    # =========================================================================
    ("thermochemistry", "field", "Heat in chemical reactions"),
    ("enthalpy", "physical_quantity", "Heat content"),
    ("entropy_chem", "physical_quantity", "Disorder measure"),
    ("gibbs_energy", "physical_quantity", "Available energy"),
    ("activation_energy", "energy", "Energy barrier for reaction"),
    # Kinetics
    ("reaction_kinetics", "field", "Study of reaction rates"),
    ("reaction_rate", "physical_quantity", "Speed of reaction"),
    ("rate_law", "physical_law", "Mathematical rate expression"),
    ("rate_constant", "physical_quantity", "k in rate law"),
    ("reaction_order", "property", "Exponent in rate law"),
    ("zero_order", "reaction_order", "Rate independent of concentration"),
    ("first_order", "reaction_order", "Rate proportional to concentration"),
    ("second_order", "reaction_order", "Rate proportional to [A]²"),
    ("half_life", "physical_quantity", "Time for half to react"),
    # Equilibrium
    ("chemical_equilibrium", "state", "Forward and reverse rates equal"),
    ("equilibrium_constant", "physical_quantity", "K - ratio at equilibrium"),
    ("le_chateliers_principle", "physical_law", "System opposes change"),
    # Electrochemistry
    ("electrochemistry", "field", "Electricity and chemistry"),
    ("electrolysis", "process", "Using electricity for reaction"),
    ("electrochemical_cell", "device", "Converts chemical to electrical"),
    ("galvanic_cell", "electrochemical_cell", "Spontaneous reaction cell"),
    ("electrolytic_cell", "electrochemical_cell", "Non-spontaneous reaction cell"),
    ("electrode", "physical_object", "Conducts electrons into solution"),
    ("anode", "electrode", "Oxidation electrode"),
    ("cathode", "electrode", "Reduction electrode"),
    ("standard_potential", "physical_quantity", "E° - standard electrode potential"),
    # =========================================================================
    # BIOCHEMISTRY BASICS
    # =========================================================================
    ("biomolecule", "organic_compound", "Molecule in living organisms"),
    # Carbohydrates
    ("carbohydrate", "biomolecule", "Sugar and starch molecules"),
    ("monosaccharide", "carbohydrate", "Simple sugar"),
    ("disaccharide", "carbohydrate", "Two sugars joined"),
    ("polysaccharide", "carbohydrate", "Many sugars joined"),
    ("glucose", "monosaccharide", "C6H12O6 - blood sugar"),
    ("fructose", "monosaccharide", "Fruit sugar"),
    ("galactose", "monosaccharide", "Milk sugar component"),
    ("sucrose", "disaccharide", "Table sugar"),
    ("lactose", "disaccharide", "Milk sugar"),
    ("maltose", "disaccharide", "Malt sugar"),
    ("starch", "polysaccharide", "Plant energy storage"),
    ("glycogen", "polysaccharide", "Animal energy storage"),
    ("cellulose", "polysaccharide", "Plant structural polymer"),
    # Lipids
    ("lipid", "biomolecule", "Fat and oil molecules"),
    ("fatty_acid", "lipid", "Long chain carboxylic acid"),
    ("saturated_fat", "lipid", "No double bonds"),
    ("unsaturated_fat", "lipid", "Has double bonds"),
    ("triglyceride", "lipid", "Three fatty acids on glycerol"),
    ("phospholipid", "lipid", "Fatty acid with phosphate"),
    ("steroid", "lipid", "Four-ring structure"),
    ("cholesterol", "steroid", "Cell membrane component"),
    # Proteins
    ("protein", "biomolecule", "Amino acid polymer"),
    ("amino_acid", "organic_compound", "Building block of proteins"),
    ("peptide", "protein", "Short amino acid chain"),
    ("polypeptide", "protein", "Long amino acid chain"),
    ("peptide_bond", "covalent_bond", "Bond between amino acids"),
    ("primary_structure", "structure", "Amino acid sequence"),
    ("secondary_structure", "structure", "Local folding patterns"),
    ("alpha_helix", "secondary_structure", "Helical structure"),
    ("beta_sheet", "secondary_structure", "Sheet structure"),
    ("tertiary_structure", "structure", "3D protein folding"),
    ("quaternary_structure", "structure", "Multiple subunit arrangement"),
    ("enzyme", "protein", "Biological catalyst"),
    # Nucleic acids
    ("nucleic_acid", "biomolecule", "DNA and RNA"),
    ("nucleotide", "organic_compound", "Building block of nucleic acids"),
    ("dna", "nucleic_acid", "Deoxyribonucleic acid"),
    ("rna", "nucleic_acid", "Ribonucleic acid"),
    ("mrna", "rna", "Messenger RNA"),
    ("trna", "rna", "Transfer RNA"),
    ("rrna", "rna", "Ribosomal RNA"),
    ("base_pair", "structure", "Complementary nucleotide pair"),
    ("adenine", "nucleotide", "A base"),
    ("guanine", "nucleotide", "G base"),
    ("cytosine", "nucleotide", "C base"),
    ("thymine", "nucleotide", "T base - DNA only"),
    ("uracil", "nucleotide", "U base - RNA only"),
    # ATP
    ("atp", "nucleotide", "Adenosine triphosphate"),
    ("adp", "nucleotide", "Adenosine diphosphate"),
    ("amp", "nucleotide", "Adenosine monophosphate"),
]


# =============================================================================
# CHEMISTRY FACTS
# =============================================================================

CHEM_FACTS = [
    # =========================================================================
    # ELEMENT PROPERTIES
    # =========================================================================
    # Element categories
    ("hydrogen", "is_a", "nonmetal"),
    ("helium", "is_a", "noble_gas"),
    ("lithium", "is_a", "alkali_metal"),
    ("beryllium", "is_a", "alkaline_earth_metal"),
    ("boron", "is_a", "metalloid"),
    ("carbon", "is_a", "nonmetal"),
    ("nitrogen", "is_a", "nonmetal"),
    ("oxygen", "is_a", "nonmetal"),
    ("fluorine", "is_a", "halogen"),
    ("neon", "is_a", "noble_gas"),
    ("sodium", "is_a", "alkali_metal"),
    ("magnesium", "is_a", "alkaline_earth_metal"),
    ("aluminum", "is_a", "post_transition_metal"),
    ("silicon", "is_a", "metalloid"),
    ("phosphorus", "is_a", "nonmetal"),
    ("sulfur", "is_a", "nonmetal"),
    ("chlorine", "is_a", "halogen"),
    ("argon", "is_a", "noble_gas"),
    ("potassium", "is_a", "alkali_metal"),
    ("calcium", "is_a", "alkaline_earth_metal"),
    ("iron", "is_a", "transition_metal"),
    ("copper", "is_a", "transition_metal"),
    ("zinc", "is_a", "transition_metal"),
    ("silver", "is_a", "transition_metal"),
    ("gold", "is_a", "transition_metal"),
    ("platinum", "is_a", "transition_metal"),
    ("mercury", "is_a", "transition_metal"),
    ("lead", "is_a", "post_transition_metal"),
    ("uranium", "is_a", "actinide"),
    # Atomic numbers
    ("hydrogen", "atomic_number", "1"),
    ("carbon", "atomic_number", "6"),
    ("nitrogen", "atomic_number", "7"),
    ("oxygen", "atomic_number", "8"),
    ("iron", "atomic_number", "26"),
    ("gold", "atomic_number", "79"),
    # =========================================================================
    # BONDING RULES
    # =========================================================================
    ("ionic_bond", "forms_between", "metal_and_nonmetal"),
    ("covalent_bond", "forms_between", "nonmetals"),
    ("metallic_bond", "forms_between", "metals"),
    ("hydrogen_bond", "stronger_than", "van_der_waals"),
    ("covalent_bond", "stronger_than", "hydrogen_bond"),
    ("ionic_bond", "stronger_than", "covalent_bond"),
    ("electronegativity", "determines", "bond_polarity"),
    ("fluorine", "has_highest", "electronegativity"),
    # =========================================================================
    # MOLECULAR GEOMETRY
    # =========================================================================
    ("methane", "has_geometry", "tetrahedral"),
    ("water", "has_geometry", "bent_geometry"),
    ("carbon_dioxide", "has_geometry", "linear_geometry"),
    ("ammonia", "has_geometry", "trigonal_pyramidal"),
    ("benzene", "has_geometry", "planar"),
    ("sp3_hybrid", "produces", "tetrahedral"),
    ("sp2_hybrid", "produces", "trigonal_planar"),
    ("sp_hybrid", "produces", "linear_geometry"),
    # =========================================================================
    # REACTION TYPES
    # =========================================================================
    ("combustion", "requires", "oxygen"),
    ("combustion", "produces", "heat"),
    ("combustion", "is_a", "exothermic"),
    ("neutralization", "produces", "water"),
    ("neutralization", "produces", "salt"),
    ("oxidation", "involves", "electron_loss"),
    ("reduction", "involves", "electron_gain"),
    ("oxidation", "coupled_with", "reduction"),
    # =========================================================================
    # ACIDS AND BASES
    # =========================================================================
    ("strong_acid", "completely", "dissociates"),
    ("weak_acid", "partially", "dissociates"),
    ("hydrochloric_acid", "formula", "HCl"),
    ("sulfuric_acid", "formula", "H2SO4"),
    ("nitric_acid", "formula", "HNO3"),
    ("acetic_acid", "formula", "CH3COOH"),
    ("sodium_hydroxide", "formula", "NaOH"),
    ("ph_7", "is", "neutral"),
    ("ph_less_than_7", "is", "acidic"),
    ("ph_greater_than_7", "is", "basic"),
    # =========================================================================
    # ORGANIC CHEMISTRY
    # =========================================================================
    # Hydrocarbon properties
    ("alkane", "has_formula", "CnH2n+2"),
    ("alkene", "has_formula", "CnH2n"),
    ("alkyne", "has_formula", "CnH2n-2"),
    ("alkane", "has_only", "single_bond"),
    ("alkene", "contains", "double_bond"),
    ("alkyne", "contains", "triple_bond"),
    ("methane", "formula", "CH4"),
    ("ethane", "formula", "C2H6"),
    ("ethene", "formula", "C2H4"),
    ("ethyne", "formula", "C2H2"),
    ("benzene", "formula", "C6H6"),
    # Functional groups
    ("alcohol", "contains", "hydroxyl_group"),
    ("aldehyde", "contains", "aldehyde_group"),
    ("ketone", "contains", "ketone_group"),
    ("carboxylic_acid", "contains", "carboxyl_group"),
    ("amine", "contains", "amino_group"),
    ("ester", "contains", "ester_group"),
    ("ethanol", "formula", "C2H5OH"),
    ("acetone", "formula", "CH3COCH3"),
    ("acetic_acid", "formula", "CH3COOH"),
    # =========================================================================
    # PHYSICAL CHEMISTRY
    # =========================================================================
    ("exothermic", "releases", "heat"),
    ("endothermic", "absorbs", "heat"),
    ("catalyst", "lowers", "activation_energy"),
    ("catalyst", "not_consumed_in", "reaction"),
    ("enzyme", "is_a", "biological_catalyst"),
    ("temperature_increase", "increases", "reaction_rate"),
    ("concentration_increase", "increases", "reaction_rate"),
    ("le_chateliers_principle", "states", "system_opposes_change"),
    # Thermodynamics
    ("negative_delta_g", "indicates", "spontaneous"),
    ("positive_delta_g", "indicates", "non_spontaneous"),
    ("gibbs_energy", "equals", "enthalpy_minus_t_times_entropy"),
    # =========================================================================
    # ELECTROCHEMISTRY
    # =========================================================================
    ("anode", "site_of", "oxidation"),
    ("cathode", "site_of", "reduction"),
    ("galvanic_cell", "is_a", "spontaneous"),
    ("electrolytic_cell", "requires", "external_power"),
    # =========================================================================
    # BIOCHEMISTRY
    # =========================================================================
    # Carbohydrates
    ("glucose", "formula", "C6H12O6"),
    ("glucose", "is_a", "monosaccharide"),
    ("sucrose", "composed_of", "glucose"),
    ("sucrose", "composed_of", "fructose"),
    ("starch", "polymer_of", "glucose"),
    ("cellulose", "polymer_of", "glucose"),
    # Proteins
    ("protein", "polymer_of", "amino_acid"),
    ("peptide_bond", "joins", "amino_acids"),
    ("enzyme", "is_a", "protein"),
    ("enzyme", "catalyzes", "biochemical_reaction"),
    # Nucleic acids
    ("dna", "contains", "adenine"),
    ("dna", "contains", "guanine"),
    ("dna", "contains", "cytosine"),
    ("dna", "contains", "thymine"),
    ("rna", "contains", "uracil"),
    ("rna", "does_not_contain", "thymine"),
    ("adenine", "pairs_with", "thymine"),
    ("guanine", "pairs_with", "cytosine"),
    ("dna", "has_structure", "double_helix"),
    ("dna", "stores", "genetic_information"),
    # ATP
    ("atp", "is_a", "energy_currency"),
    ("atp", "releases_energy_to", "adp"),
    ("atp", "contains", "three_phosphate"),
    # =========================================================================
    # IMPORTANT LAWS
    # =========================================================================
    ("law_of_conservation_of_mass", "states", "mass_conserved_in_reaction"),
    ("law_of_definite_proportions", "states", "fixed_ratio_of_elements"),
    ("law_of_multiple_proportions", "states", "small_whole_number_ratios"),
    ("avogadros_law", "states", "equal_volumes_equal_molecules"),
    ("ideal_gas_law", "formula", "PV=nRT"),
]


# =============================================================================
# LOADER
# =============================================================================


def load_chemistry_into_core(core, agent_id: str = None) -> int:
    """Load chemistry knowledge into Dorian Core."""

    if agent_id is None:
        chem_agent = core.register_agent(
            "chemistry_loader", domain="chemistry", can_verify=True
        )
        agent_id = chem_agent.agent_id

    count = 0

    print(f"  Loading {len(CHEM_CATEGORIES)} chemistry categories...")
    for name, parent, description in CHEM_CATEGORIES:
        if core.ontology:
            from dorian_ontology import Category

            parent_cat = core.ontology.categories.get(parent)
            parent_level = parent_cat.level if parent_cat else 3
            core.ontology._add_category(
                Category(
                    name=name,
                    description=description,
                    parent=parent,
                    domain="chemistry",
                    level=parent_level + 1,
                )
            )

        result = core.write(
            name,
            "subtype_of",
            parent,
            agent_id,
            source="chemistry_ontology",
            check_contradictions=False,
        )
        if result.success:
            count += 1

    print(f"  Loading {len(CHEM_FACTS)} chemistry facts...")
    for s, p, o in CHEM_FACTS:
        result = core.write(
            s, p, o, agent_id, source="chemistry_knowledge", check_contradictions=False
        )
        if result.success:
            count += 1

    print(f"  Total: {count} chemistry facts loaded")
    return count


if __name__ == "__main__":
    print("═" * 60)
    print("DORIAN CHEMISTRY DOMAIN")
    print("═" * 60)
    print(f"\nCategories: {len(CHEM_CATEGORIES)}")
    print(f"Facts: {len(CHEM_FACTS)}")
