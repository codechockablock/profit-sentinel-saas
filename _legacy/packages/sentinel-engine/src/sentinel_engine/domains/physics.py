"""
DORIAN PHYSICS DOMAIN
=====================

Comprehensive physics knowledge for Dorian Core.

Includes:
1. Extended category hierarchy (mechanics, electromagnetism, thermodynamics, quantum, relativity)
2. Physical relations (causes, conserved_in, proportional_to, etc.)
3. Fundamental laws and equations
4. Particles and forces
5. Physical constants and units

Structure:
- Categories: Types of physical objects and phenomena
- Relations: How physical quantities relate
- Facts: Laws, constants, relationships

Author: Joseph + Claude
Date: 2026-01-25
"""

from typing import Dict, List, Set, Tuple

# =============================================================================
# PHYSICS CATEGORIES
# =============================================================================

PHYSICS_CATEGORIES = [
    # =========================================================================
    # FUNDAMENTAL CONCEPTS
    # =========================================================================
    # Physical quantities (extending existing)
    ("scalar_quantity", "physical_quantity", "A quantity with only magnitude"),
    ("vector_quantity", "physical_quantity", "A quantity with magnitude and direction"),
    (
        "tensor_quantity",
        "physical_quantity",
        "A quantity with multiple components transforming under coordinate changes",
    ),
    # Base quantities
    ("length", "scalar_quantity", "The spatial extent of an object"),
    ("time", "scalar_quantity", "The duration of events"),
    ("temperature", "scalar_quantity", "A measure of thermal energy"),
    ("electric_current", "scalar_quantity", "The flow of electric charge"),
    ("amount_of_substance", "scalar_quantity", "The number of particles in a sample"),
    ("luminous_intensity", "scalar_quantity", "The power of light in a direction"),
    # Derived quantities
    ("velocity", "vector_quantity", "The rate of change of position"),
    ("acceleration", "vector_quantity", "The rate of change of velocity"),
    ("angular_velocity", "vector_quantity", "The rate of change of angular position"),
    (
        "angular_acceleration",
        "vector_quantity",
        "The rate of change of angular velocity",
    ),
    ("displacement", "vector_quantity", "Change in position"),
    ("position", "vector_quantity", "Location in space"),
    # Energy types
    ("kinetic_energy", "energy", "Energy of motion"),
    ("potential_energy", "energy", "Energy of position or configuration"),
    (
        "gravitational_potential_energy",
        "potential_energy",
        "Energy due to gravitational position",
    ),
    (
        "elastic_potential_energy",
        "potential_energy",
        "Energy stored in deformed objects",
    ),
    ("chemical_energy", "potential_energy", "Energy stored in chemical bonds"),
    ("nuclear_energy", "energy", "Energy stored in atomic nuclei"),
    ("thermal_energy", "energy", "Energy of random molecular motion"),
    ("electromagnetic_energy", "energy", "Energy carried by electromagnetic fields"),
    ("mechanical_energy", "energy", "Sum of kinetic and potential energy"),
    ("rest_energy", "energy", "Energy equivalent of rest mass"),
    # Force types
    ("contact_force", "force", "A force requiring physical contact"),
    ("field_force", "force", "A force acting at a distance"),
    ("friction", "contact_force", "Force opposing relative motion"),
    ("static_friction", "friction", "Friction preventing motion"),
    ("kinetic_friction", "friction", "Friction during motion"),
    ("normal_force", "contact_force", "Force perpendicular to a surface"),
    ("tension", "contact_force", "Force transmitted through a string or rope"),
    (
        "spring_force",
        "contact_force",
        "Force exerted by a compressed or stretched spring",
    ),
    ("drag", "contact_force", "Force opposing motion through a fluid"),
    ("buoyancy", "contact_force", "Upward force on an object in a fluid"),
    ("gravitational_force", "field_force", "Force of gravitational attraction"),
    ("electromagnetic_force", "field_force", "Force between charged particles"),
    ("electric_force", "electromagnetic_force", "Force between electric charges"),
    (
        "magnetic_force",
        "electromagnetic_force",
        "Force on moving charges in magnetic field",
    ),
    ("strong_nuclear_force", "field_force", "Force binding quarks and nucleons"),
    ("weak_nuclear_force", "field_force", "Force responsible for radioactive decay"),
    # =========================================================================
    # PARTICLES
    # =========================================================================
    # Fundamental particles
    ("elementary_particle", "particle", "A particle with no known substructure"),
    ("composite_particle", "particle", "A particle made of other particles"),
    # Fermions
    ("fermion", "elementary_particle", "A particle with half-integer spin"),
    ("lepton", "fermion", "A fermion that does not experience strong force"),
    ("quark", "fermion", "A fermion that experiences strong force"),
    # Leptons
    ("electron", "lepton", "A negatively charged lepton"),
    ("muon", "lepton", "A heavy electron-like particle"),
    ("tau", "lepton", "The heaviest lepton"),
    ("electron_neutrino", "lepton", "Neutrino associated with electron"),
    ("muon_neutrino", "lepton", "Neutrino associated with muon"),
    ("tau_neutrino", "lepton", "Neutrino associated with tau"),
    ("neutrino", "lepton", "A nearly massless neutral lepton"),
    # Quarks
    ("up_quark", "quark", "The lightest quark with charge +2/3"),
    ("down_quark", "quark", "A light quark with charge -1/3"),
    ("charm_quark", "quark", "A medium quark with charge +2/3"),
    ("strange_quark", "quark", "A medium quark with charge -1/3"),
    ("top_quark", "quark", "The heaviest quark with charge +2/3"),
    ("bottom_quark", "quark", "A heavy quark with charge -1/3"),
    # Bosons
    ("boson", "elementary_particle", "A particle with integer spin"),
    ("gauge_boson", "boson", "A boson mediating fundamental forces"),
    ("photon", "gauge_boson", "The carrier of electromagnetic force"),
    ("gluon", "gauge_boson", "The carrier of strong force"),
    ("w_boson", "gauge_boson", "A carrier of weak force"),
    ("z_boson", "gauge_boson", "A carrier of weak force"),
    ("higgs_boson", "boson", "The particle giving mass to other particles"),
    ("graviton", "gauge_boson", "The hypothetical carrier of gravity"),
    # Composite particles
    ("hadron", "composite_particle", "A particle made of quarks"),
    ("baryon", "hadron", "A hadron made of three quarks"),
    ("meson", "hadron", "A hadron made of a quark and antiquark"),
    ("proton", "baryon", "A positively charged baryon"),
    ("neutron", "baryon", "A neutral baryon"),
    ("pion", "meson", "The lightest meson"),
    ("kaon", "meson", "A meson containing a strange quark"),
    # Atoms and beyond
    ("atom", "composite_particle", "A nucleus with orbiting electrons"),
    ("ion", "atom", "An atom with net electric charge"),
    ("isotope", "atom", "Atoms with same protons but different neutrons"),
    ("nucleus", "composite_particle", "The dense core of an atom"),
    ("molecule", "composite_particle", "A group of bonded atoms"),
    # Antiparticles
    ("antiparticle", "particle", "A particle with opposite quantum numbers"),
    ("positron", "antiparticle", "The antiparticle of the electron"),
    ("antiproton", "antiparticle", "The antiparticle of the proton"),
    ("antineutron", "antiparticle", "The antiparticle of the neutron"),
    ("antimatter", "physical_object", "Matter composed of antiparticles"),
    # =========================================================================
    # FIELDS
    # =========================================================================
    ("scalar_field", "field", "A field assigning a scalar to each point"),
    ("vector_field", "field", "A field assigning a vector to each point"),
    ("tensor_field", "field", "A field assigning a tensor to each point"),
    ("electric_field", "vector_field", "The field created by electric charges"),
    ("magnetic_field", "vector_field", "The field created by moving charges"),
    ("electromagnetic_field", "field", "The unified electric and magnetic field"),
    ("gravitational_field", "vector_field", "The field created by mass"),
    ("higgs_field", "scalar_field", "The field giving particles mass"),
    ("quantum_field", "field", "A field in quantum field theory"),
    # =========================================================================
    # WAVES
    # =========================================================================
    ("mechanical_wave", "wave", "A wave requiring a medium"),
    (
        "electromagnetic_wave",
        "wave",
        "A wave of oscillating electric and magnetic fields",
    ),
    ("matter_wave", "wave", "The wave nature of particles"),
    ("sound_wave", "mechanical_wave", "A longitudinal pressure wave"),
    ("light", "electromagnetic_wave", "Visible electromagnetic radiation"),
    ("radio_wave", "electromagnetic_wave", "Long wavelength EM radiation"),
    ("microwave", "electromagnetic_wave", "EM radiation between radio and infrared"),
    ("infrared", "electromagnetic_wave", "EM radiation between microwave and visible"),
    ("ultraviolet", "electromagnetic_wave", "EM radiation beyond visible violet"),
    ("xray", "electromagnetic_wave", "High energy EM radiation"),
    ("gamma_ray", "electromagnetic_wave", "Highest energy EM radiation"),
    ("longitudinal_wave", "wave", "A wave oscillating parallel to propagation"),
    ("transverse_wave", "wave", "A wave oscillating perpendicular to propagation"),
    ("standing_wave", "wave", "A wave with stationary nodes"),
    # =========================================================================
    # THERMODYNAMICS
    # =========================================================================
    ("thermodynamic_system", "system", "A system described by thermodynamic variables"),
    (
        "isolated_system",
        "thermodynamic_system",
        "A system exchanging nothing with surroundings",
    ),
    ("closed_system", "thermodynamic_system", "A system exchanging only energy"),
    ("open_system", "thermodynamic_system", "A system exchanging energy and matter"),
    ("heat", "energy", "Energy transferred due to temperature difference"),
    ("work", "energy", "Energy transferred by force through distance"),
    ("entropy", "physical_quantity", "A measure of disorder or information"),
    ("enthalpy", "physical_quantity", "Heat content at constant pressure"),
    ("free_energy", "physical_quantity", "Energy available to do work"),
    (
        "gibbs_free_energy",
        "free_energy",
        "Free energy at constant pressure and temperature",
    ),
    (
        "helmholtz_free_energy",
        "free_energy",
        "Free energy at constant volume and temperature",
    ),
    # States of matter
    ("state_of_matter", "physical_state", "A distinct form of matter"),
    ("solid", "state_of_matter", "Matter with fixed shape and volume"),
    ("liquid", "state_of_matter", "Matter with fixed volume but no fixed shape"),
    ("gas", "state_of_matter", "Matter with no fixed shape or volume"),
    ("plasma", "state_of_matter", "Ionized gas with free electrons"),
    ("bose_einstein_condensate", "state_of_matter", "Matter at near absolute zero"),
    ("superfluid", "state_of_matter", "Fluid with zero viscosity"),
    ("superconductor", "state_of_matter", "Material with zero electrical resistance"),
    # =========================================================================
    # QUANTUM MECHANICS
    # =========================================================================
    ("quantum_state", "state", "The state of a quantum system"),
    (
        "wave_function",
        "mathematical_object",
        "Mathematical description of quantum state",
    ),
    ("superposition", "quantum_state", "A combination of multiple quantum states"),
    ("entanglement", "quantum_state", "Correlated quantum states"),
    ("quantum_number", "number", "A number characterizing quantum states"),
    ("spin", "quantum_number", "Intrinsic angular momentum"),
    ("orbital", "quantum_state", "The probability distribution of an electron"),
    ("energy_level", "quantum_state", "A discrete energy value"),
    ("ground_state", "energy_level", "The lowest energy state"),
    ("excited_state", "energy_level", "A state above ground state"),
    # =========================================================================
    # RELATIVITY
    # =========================================================================
    (
        "spacetime",
        "mathematical_structure",
        "The four-dimensional fabric of the universe",
    ),
    (
        "reference_frame",
        "mathematical_structure",
        "A coordinate system for observation",
    ),
    ("inertial_frame", "reference_frame", "A non-accelerating reference frame"),
    ("proper_time", "time", "Time measured in the rest frame"),
    ("coordinate_time", "time", "Time measured in a moving frame"),
    ("time_dilation", "physical_process", "The slowing of time in moving frames"),
    (
        "length_contraction",
        "physical_process",
        "The shortening of length in moving frames",
    ),
    (
        "lorentz_transformation",
        "mathematical_structure",
        "Transformation between inertial frames",
    ),
    ("black_hole", "physical_object", "A region where gravity prevents escape"),
    ("event_horizon", "region", "The boundary of a black hole"),
    ("singularity", "point", "A point of infinite density"),
    ("gravitational_wave", "wave", "A ripple in spacetime"),
    ("dark_matter", "physical_object", "Invisible matter detected by gravity"),
    ("dark_energy", "energy", "Energy causing universe expansion"),
    # =========================================================================
    # PHYSICAL LAWS (as categories)
    # =========================================================================
    ("conservation_law", "physical_law", "A law stating a quantity is conserved"),
    ("equation_of_motion", "physical_law", "An equation describing how systems evolve"),
    (
        "constitutive_equation",
        "physical_law",
        "An equation relating material properties",
    ),
    ("thermodynamic_law", "physical_law", "A law governing heat and energy"),
    # =========================================================================
    # PHYSICAL PROCESSES
    # =========================================================================
    ("decay", "physical_process", "The transformation of a particle into others"),
    ("radioactive_decay", "decay", "The decay of unstable nuclei"),
    ("alpha_decay", "radioactive_decay", "Emission of an alpha particle"),
    ("beta_decay", "radioactive_decay", "Emission of a beta particle"),
    ("gamma_decay", "radioactive_decay", "Emission of gamma radiation"),
    ("fission", "physical_process", "The splitting of a heavy nucleus"),
    ("fusion", "physical_process", "The combining of light nuclei"),
    ("annihilation", "physical_process", "The destruction of matter-antimatter pairs"),
    ("scattering", "physical_process", "The deflection of particles"),
    ("absorption", "physical_process", "The taking in of energy or particles"),
    ("emission", "physical_process", "The release of energy or particles"),
    ("reflection", "physical_process", "The bouncing of waves off a surface"),
    ("refraction", "physical_process", "The bending of waves at a boundary"),
    ("diffraction", "physical_process", "The spreading of waves around obstacles"),
    ("interference", "physical_process", "The combination of waves"),
    ("polarization", "physical_process", "The alignment of wave oscillations"),
    ("ionization", "physical_process", "The removal of electrons from atoms"),
    ("phase_transition", "physical_process", "The change between states of matter"),
]


# =============================================================================
# PHYSICS RELATIONS
# =============================================================================

PHYSICS_RELATIONS = [
    # Conservation relations
    (
        "conserved_in",
        "X conserved_in Y means quantity X is constant in system Y",
        {"asymmetric"},
    ),
    (
        "conserved_when",
        "X conserved_when Y means X is conserved under condition Y",
        {"asymmetric"},
    ),
    # Proportionality
    (
        "proportional_to",
        "X proportional_to Y means X = kY for some constant k",
        {"asymmetric"},
    ),
    (
        "inversely_proportional_to",
        "X inversely_proportional_to Y means X = k/Y",
        {"asymmetric"},
    ),
    (
        "squared_proportional_to",
        "X squared_proportional_to Y means X ∝ Y²",
        {"asymmetric"},
    ),
    ("inverse_squared_to", "X inverse_squared_to Y means X ∝ 1/Y²", {"asymmetric"}),
    # Physical relationships
    ("mediates", "X mediates Y means X carries force Y", {"asymmetric"}),
    ("emits", "X emits Y means X produces Y", {"asymmetric"}),
    ("absorbs", "X absorbs Y means X takes in Y", {"asymmetric"}),
    ("decays_into", "X decays_into Y means X transforms into Y", {"asymmetric"}),
    ("composed_of", "X composed_of Y means X is made of Y", {"asymmetric"}),
    ("interacts_via", "X interacts_via Y means X experiences force Y", {"asymmetric"}),
    ("transforms_into", "X transforms_into Y means X becomes Y", {"asymmetric"}),
    # Equations
    (
        "equals",
        "X equals Y means X = Y mathematically",
        {"symmetric", "transitive", "reflexive"},
    ),
    ("defined_as", "X defined_as Y means X is defined by expression Y", {"asymmetric"}),
    ("measured_in", "X measured_in Y means X has units Y", {"asymmetric"}),
    # Physical constraints
    ("bounded_by", "X bounded_by Y means Y is a limit for X", {"asymmetric"}),
    ("approaches", "X approaches Y means X tends toward Y", {"asymmetric"}),
    (
        "quantized_in",
        "X quantized_in Y means X comes in discrete units of Y",
        {"asymmetric"},
    ),
]


# =============================================================================
# PHYSICS FACTS
# =============================================================================

PHYSICS_FACTS = [
    # =========================================================================
    # FUNDAMENTAL FORCES
    # =========================================================================
    # The four forces
    ("gravitational_force", "is_a", "fundamental_force"),
    ("electromagnetic_force", "is_a", "fundamental_force"),
    ("strong_nuclear_force", "is_a", "fundamental_force"),
    ("weak_nuclear_force", "is_a", "fundamental_force"),
    # Force carriers
    ("photon", "mediates", "electromagnetic_force"),
    ("gluon", "mediates", "strong_nuclear_force"),
    ("w_boson", "mediates", "weak_nuclear_force"),
    ("z_boson", "mediates", "weak_nuclear_force"),
    ("graviton", "mediates", "gravitational_force"),
    # Force ranges
    ("gravitational_force", "has_range", "infinite"),
    ("electromagnetic_force", "has_range", "infinite"),
    ("strong_nuclear_force", "has_range", "subatomic"),
    ("weak_nuclear_force", "has_range", "subatomic"),
    # Force strengths (relative)
    ("strong_nuclear_force", "stronger_than", "electromagnetic_force"),
    ("electromagnetic_force", "stronger_than", "weak_nuclear_force"),
    ("weak_nuclear_force", "stronger_than", "gravitational_force"),
    # =========================================================================
    # PARTICLE PROPERTIES
    # =========================================================================
    # Electron
    ("electron", "has_charge", "negative"),
    ("electron", "has_spin", "one_half"),
    ("electron", "has_mass", "0.511_MeV"),
    ("electron", "is_a", "lepton"),
    ("electron", "interacts_via", "electromagnetic_force"),
    ("electron", "interacts_via", "weak_nuclear_force"),
    # Proton
    ("proton", "has_charge", "positive"),
    ("proton", "composed_of", "up_quark"),
    ("proton", "composed_of", "down_quark"),
    ("proton", "has_mass", "938.3_MeV"),
    ("proton", "interacts_via", "strong_nuclear_force"),
    ("proton", "interacts_via", "electromagnetic_force"),
    # Neutron
    ("neutron", "has_charge", "neutral"),
    ("neutron", "composed_of", "up_quark"),
    ("neutron", "composed_of", "down_quark"),
    ("neutron", "has_mass", "939.6_MeV"),
    ("neutron", "decays_into", "proton"),
    ("neutron", "interacts_via", "strong_nuclear_force"),
    # Photon
    ("photon", "has_charge", "neutral"),
    ("photon", "has_mass", "zero"),
    ("photon", "has_spin", "one"),
    ("photon", "travels_at", "speed_of_light"),
    # Neutrinos
    ("neutrino", "has_charge", "neutral"),
    ("neutrino", "has_mass", "nearly_zero"),
    ("neutrino", "interacts_via", "weak_nuclear_force"),
    # =========================================================================
    # CONSERVATION LAWS
    # =========================================================================
    # Energy
    ("energy", "conserved_in", "isolated_system"),
    ("total_energy", "conserved_in", "all_interactions"),
    ("mechanical_energy", "conserved_when", "no_friction"),
    # Momentum
    ("momentum", "conserved_in", "isolated_system"),
    ("momentum", "conserved_in", "all_collisions"),
    ("angular_momentum", "conserved_in", "isolated_system"),
    # Charge
    ("electric_charge", "conserved_in", "all_interactions"),
    ("color_charge", "conserved_in", "strong_interactions"),
    # Other conserved quantities
    ("baryon_number", "conserved_in", "all_interactions"),
    ("lepton_number", "conserved_in", "all_interactions"),
    # =========================================================================
    # NEWTON'S LAWS
    # =========================================================================
    ("newtons_first_law", "is_a", "physical_law"),
    ("newtons_first_law", "states", "objects_at_rest_stay_at_rest"),
    ("newtons_first_law", "also_called", "law_of_inertia"),
    ("newtons_second_law", "is_a", "physical_law"),
    ("newtons_second_law", "states", "f_equals_ma"),
    ("force", "equals", "mass_times_acceleration"),
    ("acceleration", "proportional_to", "force"),
    ("acceleration", "inversely_proportional_to", "mass"),
    ("newtons_third_law", "is_a", "physical_law"),
    ("newtons_third_law", "states", "action_equals_reaction"),
    ("law_of_gravitation", "is_a", "physical_law"),
    ("gravitational_force", "proportional_to", "mass"),
    ("gravitational_force", "inverse_squared_to", "distance"),
    # =========================================================================
    # THERMODYNAMICS LAWS
    # =========================================================================
    ("zeroth_law_of_thermodynamics", "is_a", "thermodynamic_law"),
    ("zeroth_law_of_thermodynamics", "states", "thermal_equilibrium_is_transitive"),
    ("first_law_of_thermodynamics", "is_a", "thermodynamic_law"),
    ("first_law_of_thermodynamics", "states", "energy_is_conserved"),
    ("first_law_of_thermodynamics", "also_called", "conservation_of_energy"),
    ("second_law_of_thermodynamics", "is_a", "thermodynamic_law"),
    ("second_law_of_thermodynamics", "states", "entropy_increases"),
    ("entropy", "increases_in", "isolated_system"),
    ("heat", "flows_from", "hot_to_cold"),
    ("third_law_of_thermodynamics", "is_a", "thermodynamic_law"),
    ("third_law_of_thermodynamics", "states", "absolute_zero_unattainable"),
    ("entropy", "approaches", "zero_at_absolute_zero"),
    # =========================================================================
    # ELECTROMAGNETISM
    # =========================================================================
    ("maxwells_equations", "is_a", "physical_law"),
    ("maxwells_equations", "describes", "electromagnetic_field"),
    ("maxwells_equations", "unifies", "electric_field"),
    ("maxwells_equations", "unifies", "magnetic_field"),
    ("coulombs_law", "is_a", "physical_law"),
    ("coulombs_law", "describes", "electric_force"),
    ("electric_force", "proportional_to", "charge"),
    ("electric_force", "inverse_squared_to", "distance"),
    ("faradays_law", "is_a", "physical_law"),
    ("faradays_law", "states", "changing_magnetic_field_induces_electric_field"),
    ("amperes_law", "is_a", "physical_law"),
    ("amperes_law", "states", "current_creates_magnetic_field"),
    ("ohms_law", "is_a", "physical_law"),
    ("ohms_law", "states", "v_equals_ir"),
    ("voltage", "equals", "current_times_resistance"),
    # Light
    ("light", "is_a", "electromagnetic_wave"),
    ("light", "has_property", "wave_particle_duality"),
    ("light", "travels_at", "speed_of_light"),
    ("speed_of_light", "equals", "299792458_m_per_s"),
    ("speed_of_light", "is_a", "physical_constant"),
    ("speed_of_light", "is_a", "maximum_speed"),
    # =========================================================================
    # QUANTUM MECHANICS
    # =========================================================================
    ("schrodinger_equation", "is_a", "equation_of_motion"),
    ("schrodinger_equation", "describes", "wave_function"),
    ("schrodinger_equation", "fundamental_to", "quantum_mechanics"),
    ("heisenberg_uncertainty_principle", "is_a", "physical_law"),
    ("heisenberg_uncertainty_principle", "states", "position_momentum_uncertainty"),
    ("position", "uncertainty_related_to", "momentum"),
    ("energy", "uncertainty_related_to", "time"),
    ("pauli_exclusion_principle", "is_a", "physical_law"),
    ("pauli_exclusion_principle", "states", "no_two_fermions_same_state"),
    ("pauli_exclusion_principle", "applies_to", "fermion"),
    ("wave_particle_duality", "is_a", "physical_law"),
    ("wave_particle_duality", "applies_to", "photon"),
    ("wave_particle_duality", "applies_to", "electron"),
    ("wave_particle_duality", "applies_to", "all_particles"),
    # Planck
    ("planck_constant", "is_a", "physical_constant"),
    ("planck_constant", "equals", "6.626e-34_J_s"),
    ("energy", "quantized_in", "planck_constant"),
    ("photon_energy", "equals", "planck_constant_times_frequency"),
    # =========================================================================
    # RELATIVITY
    # =========================================================================
    # Special relativity
    ("special_relativity", "is_a", "theory"),
    ("special_relativity", "proposed_by", "einstein"),
    ("special_relativity", "states", "speed_of_light_constant"),
    ("special_relativity", "states", "time_dilation_occurs"),
    ("special_relativity", "states", "length_contraction_occurs"),
    ("mass_energy_equivalence", "is_a", "physical_law"),
    ("mass_energy_equivalence", "states", "e_equals_mc_squared"),
    ("energy", "equals", "mass_times_c_squared"),
    ("rest_energy", "proportional_to", "mass"),
    # General relativity
    ("general_relativity", "is_a", "theory"),
    ("general_relativity", "proposed_by", "einstein"),
    ("general_relativity", "describes", "gravity"),
    ("general_relativity", "states", "gravity_is_spacetime_curvature"),
    ("mass", "curves", "spacetime"),
    ("spacetime_curvature", "causes", "gravity"),
    ("einsteins_field_equations", "is_a", "equation_of_motion"),
    ("einsteins_field_equations", "describes", "spacetime"),
    ("einsteins_field_equations", "fundamental_to", "general_relativity"),
    # =========================================================================
    # PHYSICAL CONSTANTS
    # =========================================================================
    ("speed_of_light", "symbol", "c"),
    ("planck_constant", "symbol", "h"),
    ("gravitational_constant", "symbol", "G"),
    ("boltzmann_constant", "symbol", "k"),
    ("elementary_charge", "symbol", "e"),
    ("electron_mass", "symbol", "m_e"),
    ("proton_mass", "symbol", "m_p"),
    ("avogadro_number", "symbol", "N_A"),
    ("permittivity_of_free_space", "symbol", "epsilon_0"),
    ("permeability_of_free_space", "symbol", "mu_0"),
    ("gravitational_constant", "is_a", "physical_constant"),
    ("gravitational_constant", "equals", "6.674e-11_N_m2_kg2"),
    ("boltzmann_constant", "is_a", "physical_constant"),
    ("boltzmann_constant", "equals", "1.381e-23_J_per_K"),
    ("elementary_charge", "is_a", "physical_constant"),
    ("elementary_charge", "equals", "1.602e-19_C"),
    ("avogadro_number", "is_a", "physical_constant"),
    ("avogadro_number", "equals", "6.022e23_per_mol"),
    # =========================================================================
    # IMPORTANT EQUATIONS
    # =========================================================================
    # Mechanics
    ("kinetic_energy", "equals", "half_mass_velocity_squared"),
    ("gravitational_potential_energy", "equals", "mass_times_g_times_height"),
    ("momentum", "equals", "mass_times_velocity"),
    ("work", "equals", "force_times_distance"),
    ("power", "equals", "work_per_time"),
    # Waves
    ("wave_speed", "equals", "frequency_times_wavelength"),
    ("photon_energy", "equals", "h_times_frequency"),
    ("de_broglie_wavelength", "equals", "h_over_momentum"),
    # Thermodynamics
    ("ideal_gas_law", "states", "pv_equals_nrt"),
    ("pressure_times_volume", "equals", "n_times_r_times_temperature"),
    # =========================================================================
    # ATOMIC STRUCTURE
    # =========================================================================
    ("atom", "composed_of", "nucleus"),
    ("atom", "composed_of", "electron"),
    ("nucleus", "composed_of", "proton"),
    ("nucleus", "composed_of", "neutron"),
    ("atomic_number", "equals", "number_of_protons"),
    ("mass_number", "equals", "protons_plus_neutrons"),
    ("electron", "orbits", "nucleus"),
    ("electron", "bound_by", "electromagnetic_force"),
    # =========================================================================
    # COSMOLOGY
    # =========================================================================
    ("big_bang", "is_a", "theory"),
    ("big_bang", "describes", "origin_of_universe"),
    ("universe", "expanding", "true"),
    ("hubbles_law", "states", "velocity_proportional_to_distance"),
    ("cosmic_microwave_background", "is_a", "electromagnetic_radiation"),
    ("cosmic_microwave_background", "evidence_for", "big_bang"),
    ("dark_matter", "constitutes", "27_percent_of_universe"),
    ("dark_energy", "constitutes", "68_percent_of_universe"),
    ("ordinary_matter", "constitutes", "5_percent_of_universe"),
    # =========================================================================
    # STANDARD MODEL
    # =========================================================================
    ("standard_model", "is_a", "theory"),
    ("standard_model", "describes", "elementary_particle"),
    ("standard_model", "includes", "quark"),
    ("standard_model", "includes", "lepton"),
    ("standard_model", "includes", "gauge_boson"),
    ("standard_model", "includes", "higgs_boson"),
    ("standard_model", "does_not_include", "graviton"),
    ("standard_model", "does_not_explain", "dark_matter"),
    ("standard_model", "does_not_explain", "dark_energy"),
]


# =============================================================================
# PHYSICS INFERENCE RULES
# =============================================================================


def get_physics_inference_rules():
    """Return domain-specific inference rules for physics."""

    def conservation_inference(facts):
        """If X conserved_in Y and system is Y, then X is constant."""
        inferred = []

        conserved = {}
        for f in facts:
            if f.predicate == "conserved_in":
                if f.subject not in conserved:
                    conserved[f.subject] = []
                conserved[f.subject].append(f.object)

        system_types = {}
        for f in facts:
            if f.predicate == "is_a":
                if f.subject not in system_types:
                    system_types[f.subject] = []
                system_types[f.subject].append(f.object)

        # If a system is an isolated_system, and X conserved_in isolated_system, then X constant in that system
        for system, types in system_types.items():
            for t in types:
                for quantity, systems in conserved.items():
                    if t in systems:
                        inferred.append(
                            (
                                quantity,
                                "constant_in",
                                system,
                                f"Conservation: {quantity} conserved_in {t}, {system} is_a {t}",
                            )
                        )

        return inferred

    def force_carrier_inference(facts):
        """If X mediates Y and A interacts_via Y, then A can exchange X."""
        inferred = []

        mediators = {}
        for f in facts:
            if f.predicate == "mediates":
                mediators[f.object] = f.subject  # force -> carrier

        for f in facts:
            if f.predicate == "interacts_via":
                particle = f.subject
                force = f.object
                if force in mediators:
                    carrier = mediators[force]
                    inferred.append(
                        (
                            particle,
                            "can_exchange",
                            carrier,
                            f"Force carrier: {particle} interacts_via {force}, {carrier} mediates {force}",
                        )
                    )

        return inferred

    def proportionality_chain(facts):
        """If A proportional_to B and B proportional_to C, then A proportional_to C."""
        inferred = []

        prop = {}
        for f in facts:
            if f.predicate == "proportional_to":
                if f.subject not in prop:
                    prop[f.subject] = []
                prop[f.subject].append(f.object)

        # Transitive closure
        for a, bs in prop.items():
            for b in bs:
                if b in prop:
                    for c in prop[b]:
                        if c not in bs and c != a:
                            inferred.append(
                                (
                                    a,
                                    "proportional_to",
                                    c,
                                    f"Transitivity: {a} ∝ {b}, {b} ∝ {c}",
                                )
                            )

        return inferred

    return [
        ("physics", conservation_inference),
        ("physics", force_carrier_inference),
        ("physics", proportionality_chain),
    ]


# =============================================================================
# LOADER FUNCTION
# =============================================================================


def load_physics_into_core(core, agent_id: str = None) -> int:
    """
    Load all physics knowledge into a Dorian Core.

    Args:
        core: DorianCore instance
        agent_id: Agent ID to attribute facts to (creates one if None)

    Returns:
        Number of facts added
    """
    # Create physics agent if needed
    if agent_id is None:
        physics_agent = core.register_agent(
            "physics_loader", domain="physics", can_verify=True
        )
        agent_id = physics_agent.agent_id

    count = 0

    # Add category facts
    print(f"  Loading {len(PHYSICS_CATEGORIES)} physics categories...")
    for name, parent, description in PHYSICS_CATEGORIES:
        # Add to ontology if core has one
        if core.ontology:
            from dorian_ontology import Category

            parent_cat = core.ontology.categories.get(parent)
            parent_level = parent_cat.level if parent_cat else 3
            core.ontology._add_category(
                Category(
                    name=name,
                    description=description,
                    parent=parent,
                    domain="physics",
                    level=parent_level + 1,
                )
            )

        # Add subtype_of fact
        result = core.write(
            name,
            "subtype_of",
            parent,
            agent_id,
            source="physics_ontology",
            check_contradictions=False,
        )
        if result.success:
            count += 1

    # Add physics facts
    print(f"  Loading {len(PHYSICS_FACTS)} physics facts...")
    for s, p, o in PHYSICS_FACTS:
        result = core.write(
            s, p, o, agent_id, source="physics_knowledge", check_contradictions=False
        )
        if result.success:
            count += 1

    # Register inference rules
    print("  Registering physics inference rules...")
    for domain, rule in get_physics_inference_rules():
        core.inference_engine.add_domain_rule(domain, rule)

    print(f"  Total: {count} physics facts loaded")
    return count


# =============================================================================
# MAIN - DEMO
# =============================================================================

if __name__ == "__main__":
    print("═" * 60)
    print("DORIAN PHYSICS DOMAIN")
    print("═" * 60)

    print(f"\nCategories: {len(PHYSICS_CATEGORIES)}")
    print(f"Relations: {len(PHYSICS_RELATIONS)}")
    print(f"Facts: {len(PHYSICS_FACTS)}")

    print("\nSample categories:")
    for name, parent, desc in PHYSICS_CATEGORIES[:10]:
        print(f"  {name} <- {parent}")

    print("\nSample facts:")
    for s, p, o in PHYSICS_FACTS[:15]:
        print(f"  {s} {p} {o}")

    print("\nForce carriers:")
    for s, p, o in PHYSICS_FACTS:
        if p == "mediates":
            print(f"  {s} {p} {o}")

    print("\nTo load into Dorian Core:")
    print("  from dorian_physics import load_physics_into_core")
    print("  count = load_physics_into_core(core)")
