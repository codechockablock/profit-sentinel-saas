"""
DORIAN BIOLOGY DOMAIN
=====================

Comprehensive biology knowledge for Dorian Core.

Includes:
1. Cell biology
2. Genetics and molecular biology
3. Evolution and taxonomy
4. Ecology
5. Human physiology
6. Microbiology

Author: Joseph + Claude
Date: 2026-01-25
"""

from typing import Dict, List, Set, Tuple

# =============================================================================
# BIOLOGY CATEGORIES
# =============================================================================

BIO_CATEGORIES = [
    # =========================================================================
    # CELL BIOLOGY
    # =========================================================================
    ("cell", "biological_structure", "Basic unit of life"),
    ("prokaryotic_cell", "cell", "Cell without nucleus"),
    ("eukaryotic_cell", "cell", "Cell with nucleus"),
    ("animal_cell", "eukaryotic_cell", "Animal cell type"),
    ("plant_cell", "eukaryotic_cell", "Plant cell type"),
    ("fungal_cell", "eukaryotic_cell", "Fungal cell type"),
    # Organelles
    ("organelle", "biological_structure", "Specialized cell structure"),
    ("nucleus", "organelle", "Contains genetic material"),
    ("nucleolus", "organelle", "Ribosome production site"),
    ("mitochondrion", "organelle", "Powerhouse of the cell"),
    ("chloroplast", "organelle", "Photosynthesis site"),
    ("ribosome", "organelle", "Protein synthesis site"),
    ("endoplasmic_reticulum", "organelle", "Protein/lipid synthesis"),
    ("rough_er", "endoplasmic_reticulum", "ER with ribosomes"),
    ("smooth_er", "endoplasmic_reticulum", "ER without ribosomes"),
    ("golgi_apparatus", "organelle", "Protein processing/packaging"),
    ("lysosome", "organelle", "Digestive organelle"),
    ("peroxisome", "organelle", "Detoxification organelle"),
    ("vacuole", "organelle", "Storage organelle"),
    ("central_vacuole", "vacuole", "Large plant cell vacuole"),
    ("centriole", "organelle", "Cell division organelle"),
    ("cytoskeleton", "organelle", "Cell structural support"),
    ("microtubule", "cytoskeleton", "Hollow protein tube"),
    ("microfilament", "cytoskeleton", "Actin filament"),
    ("intermediate_filament", "cytoskeleton", "Structural filament"),
    # Cell membrane
    ("cell_membrane", "biological_structure", "Phospholipid bilayer boundary"),
    ("plasma_membrane", "cell_membrane", "Cell outer membrane"),
    ("cell_wall", "biological_structure", "Rigid outer layer"),
    ("membrane_protein", "protein", "Protein in membrane"),
    ("channel_protein", "membrane_protein", "Forms channel through membrane"),
    ("carrier_protein", "membrane_protein", "Transports molecules"),
    ("receptor_protein", "membrane_protein", "Receives signals"),
    # Cell processes
    ("cell_process", "biological_process", "Process occurring in cells"),
    ("cell_division", "cell_process", "Cell reproduction"),
    ("mitosis", "cell_division", "Division producing identical cells"),
    ("meiosis", "cell_division", "Division producing gametes"),
    ("cytokinesis", "cell_division", "Cytoplasm division"),
    ("cell_cycle", "cell_process", "Cycle of cell growth and division"),
    ("interphase", "cell_cycle", "Growth phase"),
    ("g1_phase", "interphase", "First growth phase"),
    ("s_phase", "interphase", "DNA synthesis phase"),
    ("g2_phase", "interphase", "Second growth phase"),
    ("m_phase", "cell_cycle", "Mitotic phase"),
    ("prophase", "m_phase", "Chromosomes condense"),
    ("metaphase", "m_phase", "Chromosomes align"),
    ("anaphase", "m_phase", "Chromosomes separate"),
    ("telophase", "m_phase", "Nuclear envelopes form"),
    # =========================================================================
    # GENETICS
    # =========================================================================
    ("genetic_material", "biological_structure", "Hereditary material"),
    ("chromosome", "genetic_material", "DNA + proteins structure"),
    ("autosome", "chromosome", "Non-sex chromosome"),
    ("sex_chromosome", "chromosome", "X or Y chromosome"),
    ("homologous_chromosomes", "chromosome", "Matching chromosome pair"),
    ("sister_chromatids", "chromosome", "Identical chromosome copies"),
    ("chromatin", "genetic_material", "DNA-protein complex"),
    ("gene", "genetic_material", "Unit of heredity"),
    ("allele", "gene", "Gene variant"),
    ("dominant_allele", "allele", "Expressed when present"),
    ("recessive_allele", "allele", "Expressed when homozygous"),
    ("locus", "genetic_material", "Gene location on chromosome"),
    # Genetic concepts
    ("genotype", "property", "Genetic makeup"),
    ("phenotype", "property", "Observable characteristics"),
    ("homozygous", "genotype", "Two identical alleles"),
    ("heterozygous", "genotype", "Two different alleles"),
    ("wild_type", "allele", "Most common allele"),
    ("mutation", "genetic_material", "Change in DNA sequence"),
    ("point_mutation", "mutation", "Single nucleotide change"),
    ("insertion", "mutation", "Added nucleotides"),
    ("deletion", "mutation", "Removed nucleotides"),
    ("frameshift", "mutation", "Shifts reading frame"),
    # Inheritance patterns
    ("inheritance_pattern", "abstract", "How traits are inherited"),
    ("mendelian_inheritance", "inheritance_pattern", "Simple dominant/recessive"),
    ("codominance", "inheritance_pattern", "Both alleles expressed"),
    ("incomplete_dominance", "inheritance_pattern", "Blended expression"),
    ("sex_linked", "inheritance_pattern", "Genes on sex chromosomes"),
    ("x_linked", "sex_linked", "Genes on X chromosome"),
    ("polygenic", "inheritance_pattern", "Multiple genes for trait"),
    ("epistasis", "inheritance_pattern", "Gene interaction"),
    # Molecular genetics
    ("central_dogma", "biological_process", "DNA → RNA → Protein"),
    ("replication", "biological_process", "DNA copying"),
    ("transcription", "biological_process", "DNA to RNA"),
    ("translation", "biological_process", "RNA to protein"),
    ("genetic_code", "abstract", "Codon to amino acid mapping"),
    ("codon", "genetic_material", "Three nucleotide sequence"),
    ("start_codon", "codon", "AUG - initiates translation"),
    ("stop_codon", "codon", "Terminates translation"),
    ("promoter", "genetic_material", "Transcription start site"),
    ("terminator", "genetic_material", "Transcription stop site"),
    ("intron", "genetic_material", "Non-coding sequence"),
    ("exon", "genetic_material", "Coding sequence"),
    ("splicing", "biological_process", "Intron removal"),
    # Gene regulation
    ("gene_regulation", "biological_process", "Control of gene expression"),
    ("operon", "genetic_material", "Bacterial gene cluster"),
    ("lac_operon", "operon", "Lactose metabolism genes"),
    ("repressor", "protein", "Blocks transcription"),
    ("activator", "protein", "Enhances transcription"),
    ("enhancer", "genetic_material", "Increases transcription"),
    ("silencer", "genetic_material", "Decreases transcription"),
    ("epigenetics", "gene_regulation", "Heritable non-DNA changes"),
    ("methylation", "epigenetics", "Adding methyl groups"),
    ("histone_modification", "epigenetics", "Changing histone proteins"),
    # =========================================================================
    # EVOLUTION
    # =========================================================================
    ("evolution", "biological_process", "Change in populations over time"),
    ("natural_selection", "evolution", "Differential survival/reproduction"),
    ("artificial_selection", "evolution", "Human-directed selection"),
    ("sexual_selection", "natural_selection", "Selection for mating success"),
    ("genetic_drift", "evolution", "Random allele frequency change"),
    ("gene_flow", "evolution", "Allele transfer between populations"),
    ("speciation", "evolution", "Formation of new species"),
    ("allopatric_speciation", "speciation", "Geographic separation"),
    ("sympatric_speciation", "speciation", "Without geographic separation"),
    ("adaptive_radiation", "evolution", "Rapid diversification"),
    ("convergent_evolution", "evolution", "Similar traits, different ancestors"),
    ("divergent_evolution", "evolution", "Different traits, common ancestor"),
    ("coevolution", "evolution", "Reciprocal evolution"),
    # Evidence for evolution
    ("fossil", "evidence", "Preserved ancient organism"),
    ("homologous_structure", "evidence", "Same origin, different function"),
    ("analogous_structure", "evidence", "Different origin, same function"),
    ("vestigial_structure", "evidence", "Reduced non-functional structure"),
    # =========================================================================
    # TAXONOMY
    # =========================================================================
    ("taxon", "abstract", "Taxonomic group"),
    ("domain", "taxon", "Highest taxonomic rank"),
    ("kingdom", "taxon", "Major life division"),
    ("phylum", "taxon", "Major body plan group"),
    ("class", "taxon", "Group within phylum"),
    ("order", "taxon", "Group within class"),
    ("family", "taxon", "Group within order"),
    ("genus", "taxon", "Group of related species"),
    ("species", "taxon", "Interbreeding population"),
    # Domains
    ("bacteria", "domain", "Single-celled prokaryotes"),
    ("archaea", "domain", "Extremophile prokaryotes"),
    ("eukarya", "domain", "Organisms with nuclei"),
    # Kingdoms
    ("animalia", "kingdom", "Animals"),
    ("plantae", "kingdom", "Plants"),
    ("fungi", "kingdom", "Fungi"),
    ("protista", "kingdom", "Single-celled eukaryotes"),
    # Animal phyla
    ("chordata", "phylum", "Animals with notochord"),
    ("arthropoda", "phylum", "Jointed leg animals"),
    ("mollusca", "phylum", "Soft-bodied animals"),
    ("annelida", "phylum", "Segmented worms"),
    ("cnidaria", "phylum", "Jellyfish, coral"),
    ("porifera", "phylum", "Sponges"),
    ("echinodermata", "phylum", "Sea stars, urchins"),
    ("nematoda", "phylum", "Roundworms"),
    ("platyhelminthes", "phylum", "Flatworms"),
    # Vertebrate classes
    ("vertebrate", "chordata", "Animals with backbone"),
    ("mammalia", "vertebrate", "Mammals"),
    ("aves", "vertebrate", "Birds"),
    ("reptilia", "vertebrate", "Reptiles"),
    ("amphibia", "vertebrate", "Amphibians"),
    ("osteichthyes", "vertebrate", "Bony fish"),
    ("chondrichthyes", "vertebrate", "Cartilaginous fish"),
    # =========================================================================
    # ECOLOGY
    # =========================================================================
    ("ecology", "field", "Study of organism-environment interactions"),
    ("ecosystem", "system", "Community plus environment"),
    ("biome", "ecosystem", "Large geographic region"),
    ("community", "system", "Interacting populations"),
    ("population", "system", "Same species in an area"),
    ("habitat", "region", "Where organism lives"),
    ("niche", "abstract", "Organism's role in ecosystem"),
    # Biomes
    ("tropical_rainforest", "biome", "Hot, wet forest"),
    ("temperate_forest", "biome", "Moderate forest"),
    ("taiga", "biome", "Northern coniferous forest"),
    ("tundra", "biome", "Cold, treeless region"),
    ("grassland", "biome", "Grass-dominated region"),
    ("savanna", "grassland", "Tropical grassland"),
    ("desert", "biome", "Dry region"),
    ("freshwater", "biome", "Lakes, rivers, streams"),
    ("marine", "biome", "Ocean ecosystem"),
    ("coral_reef", "marine", "Coral-based ecosystem"),
    # Ecological relationships
    ("ecological_relationship", "relation", "Interaction between species"),
    ("symbiosis", "ecological_relationship", "Close species interaction"),
    ("mutualism", "symbiosis", "Both species benefit"),
    ("commensalism", "symbiosis", "One benefits, one unaffected"),
    ("parasitism", "symbiosis", "One benefits, one harmed"),
    ("predation", "ecological_relationship", "Predator-prey interaction"),
    ("competition", "ecological_relationship", "Contest for resources"),
    # Trophic levels
    ("trophic_level", "abstract", "Feeding level in food chain"),
    ("producer", "trophic_level", "Photosynthetic organism"),
    ("primary_consumer", "trophic_level", "Herbivore"),
    ("secondary_consumer", "trophic_level", "Carnivore eating herbivores"),
    ("tertiary_consumer", "trophic_level", "Top predator"),
    ("decomposer", "trophic_level", "Breaks down dead matter"),
    ("food_chain", "structure", "Linear feeding relationship"),
    ("food_web", "structure", "Complex feeding relationships"),
    # Cycles
    ("biogeochemical_cycle", "process", "Element cycling through ecosystem"),
    ("carbon_cycle", "biogeochemical_cycle", "Carbon movement"),
    ("nitrogen_cycle", "biogeochemical_cycle", "Nitrogen movement"),
    ("water_cycle", "biogeochemical_cycle", "Water movement"),
    ("phosphorus_cycle", "biogeochemical_cycle", "Phosphorus movement"),
    # =========================================================================
    # PHYSIOLOGY
    # =========================================================================
    ("organ_system", "biological_structure", "Group of organs with function"),
    ("organ", "biological_structure", "Tissue group with function"),
    ("tissue", "biological_structure", "Similar cells with function"),
    # Tissue types
    ("epithelial_tissue", "tissue", "Covering tissue"),
    ("connective_tissue", "tissue", "Supporting tissue"),
    ("muscle_tissue", "tissue", "Contractile tissue"),
    ("nervous_tissue", "tissue", "Signal conducting tissue"),
    # Organ systems
    ("circulatory_system", "organ_system", "Blood transport system"),
    ("respiratory_system", "organ_system", "Gas exchange system"),
    ("digestive_system", "organ_system", "Food processing system"),
    ("nervous_system", "organ_system", "Signal processing system"),
    ("endocrine_system", "organ_system", "Hormone system"),
    ("immune_system", "organ_system", "Defense system"),
    ("skeletal_system", "organ_system", "Bone support system"),
    ("muscular_system", "organ_system", "Movement system"),
    ("excretory_system", "organ_system", "Waste removal system"),
    ("reproductive_system", "organ_system", "Reproduction system"),
    ("integumentary_system", "organ_system", "Skin system"),
    ("lymphatic_system", "organ_system", "Fluid drainage system"),
    # Key organs
    ("heart", "organ", "Blood pumping organ"),
    ("lung", "organ", "Gas exchange organ"),
    ("brain", "organ", "Central nervous system organ"),
    ("liver", "organ", "Metabolic organ"),
    ("kidney", "organ", "Filtration organ"),
    ("stomach", "organ", "Digestive organ"),
    ("intestine", "organ", "Nutrient absorption organ"),
    ("small_intestine", "intestine", "Main absorption site"),
    ("large_intestine", "intestine", "Water absorption site"),
    ("pancreas", "organ", "Digestive/endocrine organ"),
    ("spleen", "organ", "Blood filtering organ"),
    ("thyroid", "organ", "Metabolism regulation"),
    # =========================================================================
    # MICROBIOLOGY
    # =========================================================================
    ("microorganism", "organism", "Microscopic organism"),
    ("bacterium", "microorganism", "Prokaryotic cell"),
    ("virus", "microorganism", "Non-cellular pathogen"),
    ("fungus", "microorganism", "Eukaryotic decomposer"),
    ("protozoan", "microorganism", "Single-celled eukaryote"),
    # Bacteria types
    ("gram_positive", "bacterium", "Thick peptidoglycan wall"),
    ("gram_negative", "bacterium", "Thin peptidoglycan wall"),
    ("coccus", "bacterium", "Spherical bacteria"),
    ("bacillus", "bacterium", "Rod-shaped bacteria"),
    ("spirillum", "bacterium", "Spiral bacteria"),
    # Virus types
    ("dna_virus", "virus", "DNA genome virus"),
    ("rna_virus", "virus", "RNA genome virus"),
    ("retrovirus", "rna_virus", "RNA to DNA virus"),
    ("bacteriophage", "virus", "Bacteria-infecting virus"),
    # Immune response
    ("immunity", "biological_process", "Defense against pathogens"),
    ("innate_immunity", "immunity", "Non-specific defense"),
    ("adaptive_immunity", "immunity", "Specific defense"),
    ("antibody", "protein", "Pathogen-binding protein"),
    ("antigen", "molecule", "Immune-triggering molecule"),
    ("lymphocyte", "cell", "Immune system cell"),
    ("t_cell", "lymphocyte", "Cell-mediated immunity"),
    ("b_cell", "lymphocyte", "Antibody-producing cell"),
    ("macrophage", "cell", "Phagocytic cell"),
    ("vaccine", "medical_treatment", "Immunity-inducing agent"),
    # =========================================================================
    # PHOTOSYNTHESIS AND RESPIRATION
    # =========================================================================
    ("photosynthesis", "biological_process", "Light to chemical energy"),
    ("light_reactions", "photosynthesis", "Light-dependent reactions"),
    ("calvin_cycle", "photosynthesis", "Carbon fixation"),
    ("photosystem", "biological_structure", "Light-capturing complex"),
    ("photosystem_i", "photosystem", "PSI"),
    ("photosystem_ii", "photosystem", "PSII"),
    ("cellular_respiration", "biological_process", "Glucose to ATP"),
    ("glycolysis", "cellular_respiration", "Glucose to pyruvate"),
    ("krebs_cycle", "cellular_respiration", "Citric acid cycle"),
    ("electron_transport_chain", "cellular_respiration", "ATP synthesis"),
    ("fermentation", "biological_process", "Anaerobic ATP production"),
    ("lactic_acid_fermentation", "fermentation", "Produces lactic acid"),
    ("alcoholic_fermentation", "fermentation", "Produces ethanol"),
]


# =============================================================================
# BIOLOGY FACTS
# =============================================================================

BIO_FACTS = [
    # =========================================================================
    # CELL BIOLOGY
    # =========================================================================
    # Cell types
    ("prokaryotic_cell", "lacks", "nucleus"),
    ("eukaryotic_cell", "has", "nucleus"),
    ("plant_cell", "has", "cell_wall"),
    ("plant_cell", "has", "chloroplast"),
    ("plant_cell", "has", "central_vacuole"),
    ("animal_cell", "lacks", "cell_wall"),
    ("animal_cell", "lacks", "chloroplast"),
    # Organelle functions
    ("mitochondrion", "produces", "atp"),
    ("mitochondrion", "called", "powerhouse_of_cell"),
    ("chloroplast", "performs", "photosynthesis"),
    ("ribosome", "performs", "protein_synthesis"),
    ("nucleus", "contains", "dna"),
    ("golgi_apparatus", "modifies", "proteins"),
    ("lysosome", "contains", "digestive_enzymes"),
    ("rough_er", "has", "ribosomes"),
    ("smooth_er", "synthesizes", "lipids"),
    # Cell membrane
    ("cell_membrane", "made_of", "phospholipid_bilayer"),
    ("cell_membrane", "is_a", "selectively_permeable"),
    # =========================================================================
    # GENETICS
    # =========================================================================
    # DNA structure
    ("dna", "composed_of", "nucleotides"),
    ("dna", "has_shape", "double_helix"),
    ("dna", "discovered_by", "watson_and_crick"),
    ("adenine", "pairs_with", "thymine"),
    ("guanine", "pairs_with", "cytosine"),
    # Central dogma
    ("dna", "transcribed_to", "rna"),
    ("rna", "translated_to", "protein"),
    ("replication", "occurs_in", "nucleus"),
    ("transcription", "occurs_in", "nucleus"),
    ("translation", "occurs_at", "ribosome"),
    # Genetic code
    ("codon", "codes_for", "amino_acid"),
    ("start_codon", "sequence", "AUG"),
    ("start_codon", "codes_for", "methionine"),
    # Inheritance
    ("dominant_allele", "masks", "recessive_allele"),
    ("homozygous", "has", "identical_alleles"),
    ("heterozygous", "has", "different_alleles"),
    # Mutations
    ("mutation", "can_cause", "genetic_disorder"),
    ("mutation", "source_of", "genetic_variation"),
    # =========================================================================
    # CELL DIVISION
    # =========================================================================
    ("mitosis", "produces", "two_identical_cells"),
    ("meiosis", "produces", "four_haploid_cells"),
    ("meiosis", "produces", "gametes"),
    ("mitosis", "maintains", "chromosome_number"),
    ("meiosis", "halves", "chromosome_number"),
    # =========================================================================
    # EVOLUTION
    # =========================================================================
    ("natural_selection", "proposed_by", "darwin"),
    ("natural_selection", "requires", "variation"),
    ("natural_selection", "requires", "heritability"),
    ("natural_selection", "requires", "differential_reproduction"),
    ("evolution", "supported_by", "fossil_record"),
    ("evolution", "supported_by", "dna_evidence"),
    ("evolution", "supported_by", "comparative_anatomy"),
    ("speciation", "requires", "reproductive_isolation"),
    # =========================================================================
    # TAXONOMY
    # =========================================================================
    # Domain classification
    ("bacteria", "is_a", "prokaryote"),
    ("archaea", "is_a", "prokaryote"),
    ("eukarya", "is_a", "eukaryote"),
    # Kingdom classification
    ("animalia", "is_a", "heterotroph"),
    ("plantae", "is_a", "autotroph"),
    ("fungi", "is_a", "heterotroph"),
    ("fungi", "performs", "decomposition"),
    # Human classification
    ("human", "species", "homo_sapiens"),
    ("human", "is_a", "mammalia"),
    ("human", "is_a", "primate"),
    # =========================================================================
    # ECOLOGY
    # =========================================================================
    # Trophic levels
    ("producer", "performs", "photosynthesis"),
    ("producer", "is_a", "autotroph"),
    ("primary_consumer", "eats", "producer"),
    ("primary_consumer", "is_a", "herbivore"),
    ("secondary_consumer", "eats", "primary_consumer"),
    ("decomposer", "breaks_down", "dead_matter"),
    # Energy flow
    ("energy", "flows", "one_direction"),
    ("energy", "lost_as", "heat"),
    ("ten_percent_rule", "states", "10_percent_energy_transfer"),
    # Relationships
    ("mutualism", "benefits", "both_species"),
    ("parasitism", "harms", "host"),
    ("predation", "controls", "prey_population"),
    # Cycles
    ("carbon_cycle", "involves", "photosynthesis"),
    ("carbon_cycle", "involves", "cellular_respiration"),
    ("nitrogen_cycle", "requires", "nitrogen_fixing_bacteria"),
    # =========================================================================
    # PHYSIOLOGY
    # =========================================================================
    # Circulatory system
    ("heart", "pumps", "blood"),
    ("artery", "carries", "oxygenated_blood"),
    ("vein", "carries", "deoxygenated_blood"),
    ("capillary", "site_of", "gas_exchange"),
    # Respiratory system
    ("lung", "performs", "gas_exchange"),
    ("oxygen", "diffuses_into", "blood"),
    ("carbon_dioxide", "diffuses_out_of", "blood"),
    # Digestive system
    ("stomach", "produces", "acid"),
    ("small_intestine", "absorbs", "nutrients"),
    ("large_intestine", "absorbs", "water"),
    ("liver", "produces", "bile"),
    ("pancreas", "produces", "digestive_enzymes"),
    # Nervous system
    ("brain", "controls", "body_functions"),
    ("neuron", "transmits", "electrical_signals"),
    ("synapse", "site_of", "neurotransmitter_release"),
    # Endocrine system
    ("hormone", "is_a", "chemical_messenger"),
    ("insulin", "lowers", "blood_glucose"),
    ("glucagon", "raises", "blood_glucose"),
    # Immune system
    ("antibody", "binds", "antigen"),
    ("t_cell", "destroys", "infected_cells"),
    ("b_cell", "produces", "antibodies"),
    ("vaccine", "triggers", "immune_response"),
    # =========================================================================
    # PHOTOSYNTHESIS AND RESPIRATION
    # =========================================================================
    # Photosynthesis
    ("photosynthesis", "equation", "6CO2+6H2O→C6H12O6+6O2"),
    ("photosynthesis", "requires", "light"),
    ("photosynthesis", "requires", "chlorophyll"),
    ("photosynthesis", "occurs_in", "chloroplast"),
    ("light_reactions", "produce", "atp"),
    ("light_reactions", "produce", "nadph"),
    ("calvin_cycle", "fixes", "carbon_dioxide"),
    # Cellular respiration
    ("cellular_respiration", "equation", "C6H12O6+6O2→6CO2+6H2O+ATP"),
    ("cellular_respiration", "occurs_in", "mitochondrion"),
    ("glycolysis", "occurs_in", "cytoplasm"),
    ("glycolysis", "produces", "2_atp"),
    ("krebs_cycle", "produces", "2_atp"),
    ("electron_transport_chain", "produces", "34_atp"),
    # Fermentation
    ("fermentation", "occurs_without", "oxygen"),
    ("lactic_acid_fermentation", "occurs_in", "muscle"),
    ("alcoholic_fermentation", "produces", "ethanol"),
]


# =============================================================================
# LOADER
# =============================================================================


def load_biology_into_core(core, agent_id: str = None) -> int:
    """Load biology knowledge into Dorian Core."""

    if agent_id is None:
        bio_agent = core.register_agent(
            "biology_loader", domain="biology", can_verify=True
        )
        agent_id = bio_agent.agent_id

    count = 0

    print(f"  Loading {len(BIO_CATEGORIES)} biology categories...")
    for name, parent, description in BIO_CATEGORIES:
        if core.ontology:
            from dorian_ontology import Category

            parent_cat = core.ontology.categories.get(parent)
            parent_level = parent_cat.level if parent_cat else 3
            core.ontology._add_category(
                Category(
                    name=name,
                    description=description,
                    parent=parent,
                    domain="biology",
                    level=parent_level + 1,
                )
            )

        result = core.write(
            name,
            "subtype_of",
            parent,
            agent_id,
            source="biology_ontology",
            check_contradictions=False,
        )
        if result.success:
            count += 1

    print(f"  Loading {len(BIO_FACTS)} biology facts...")
    for s, p, o in BIO_FACTS:
        result = core.write(
            s, p, o, agent_id, source="biology_knowledge", check_contradictions=False
        )
        if result.success:
            count += 1

    print(f"  Total: {count} biology facts loaded")
    return count


if __name__ == "__main__":
    print("═" * 60)
    print("DORIAN BIOLOGY DOMAIN")
    print("═" * 60)
    print(f"\nCategories: {len(BIO_CATEGORIES)}")
    print(f"Facts: {len(BIO_FACTS)}")
