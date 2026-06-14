"""Physiological disease progression models for digital twin simulation.

This module implements disease-specific progression models based on
differential equations and clinical knowledge. Models simulate how
patient vital signs and lab values evolve over time.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class DiseaseModel(ABC):
    """Abstract base class for disease progression models.

    All disease models must implement the evolve() method which computes
    the next patient state given current state and interventions.
    """

    @abstractmethod
    def evolve(
        self,
        current_state: Dict[str, Any],
        dt: float,
        interventions: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Compute next patient state.

        Args:
            current_state: Current patient features
            dt: Time step in hours
            interventions: Applied interventions (e.g., medications, fluids)
            rng: Random number generator for stochastic evolution

        Returns:
            Next patient state after dt hours
        """
        pass


class SepsisModel(DiseaseModel):
    """Simplified sepsis progression model.

    Based on SIRS (Systemic Inflammatory Response Syndrome) criteria
    and compartmental infection dynamics.

    Simulated variables:
        - Temperature (°C)
        - Heart rate (bpm)
        - Respiratory rate (breaths/min)
        - White blood cell count (cells/μL)
        - Blood pressure (systolic mmHg)
        - Lactate (mmol/L)

    Interventions:
        - antibiotic: Reduces infection severity
        - fluid_bolus: Increases blood pressure, dilutes lactate
        - vasopressor: Increases blood pressure

    References:
        Singer et al. (2016). The Third International Consensus Definitions
        for Sepsis and Septic Shock (Sepsis-3). JAMA.
    """

    def __init__(
        self,
        infection_growth_rate: float = 0.15,
        infection_decay_rate: float = 0.3,
        temperature_sensitivity: float = 0.5,
        hemodynamic_sensitivity: float = 0.4,
        noise_std: float = 0.05,
    ):
        """Initialize sepsis model parameters.

        Args:
            infection_growth_rate: Rate of infection worsening (per hour)
            infection_decay_rate: Rate of recovery with antibiotics
            temperature_sensitivity: How much temperature responds to infection
            hemodynamic_sensitivity: How much BP/HR respond to infection
            noise_std: Standard deviation of measurement noise
        """
        self.infection_growth_rate = infection_growth_rate
        self.infection_decay_rate = infection_decay_rate
        self.temperature_sensitivity = temperature_sensitivity
        self.hemodynamic_sensitivity = hemodynamic_sensitivity
        self.noise_std = noise_std

    def evolve(
        self,
        current_state: Dict[str, Any],
        dt: float,
        interventions: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Evolve sepsis patient state."""
        if rng is None:
            rng = np.random.RandomState()

        # Extract current vitals (with defaults)
        temp = current_state.get('temperature', 37.0)
        hr = current_state.get('heart_rate', 80)
        rr = current_state.get('respiratory_rate', 16)
        wbc = current_state.get('white_blood_cell_count', 8000)
        bp_sys = current_state.get('blood_pressure_sys', 120)
        lactate = current_state.get('lactate', 1.0)

        # Estimate infection severity from current state
        # Higher temp, HR, lactate → higher severity
        infection_severity = (
            0.3 * max(0, (temp - 37.0) / 2.0)
            + 0.3 * max(0, (hr - 80) / 40.0)
            + 0.4 * max(0, (lactate - 1.0) / 3.0)
        )
        infection_severity = np.clip(infection_severity, 0, 1)

        # Intervention effects
        antibiotic_effect = 0.0
        fluid_effect = 0.0
        vasopressor_effect = 0.0

        if interventions:
            if interventions.get('antibiotic', False):
                antibiotic_effect = self.infection_decay_rate
            if interventions.get('fluid_bolus', 0) > 0:
                # Fluid bolus in mL
                fluid_effect = interventions['fluid_bolus'] / 1000.0  # normalize
            if interventions.get('vasopressor', False):
                vasopressor_effect = 0.5

        # Update infection severity
        d_infection = (
            self.infection_growth_rate * infection_severity * (1 - infection_severity)
            - antibiotic_effect * infection_severity
        ) * dt
        infection_severity = np.clip(infection_severity + d_infection, 0, 1)

        # Temperature dynamics
        # Fever develops with infection, resolves with antibiotics
        temp_target = 37.0 + 2.5 * infection_severity
        d_temp = (temp_target - temp) * self.temperature_sensitivity * dt
        temp_new = temp + d_temp + rng.normal(0, self.noise_std * dt)
        temp_new = np.clip(temp_new, 35.0, 41.0)

        # Heart rate dynamics
        # Tachycardia with infection and fever
        hr_target = 70 + 50 * infection_severity + 10 * (temp_new - 37.0)
        d_hr = (hr_target - hr) * self.hemodynamic_sensitivity * dt
        hr_new = hr + d_hr + rng.normal(0, 5 * dt)
        hr_new = np.clip(hr_new, 50, 180)

        # Respiratory rate
        rr_target = 14 + 12 * infection_severity
        d_rr = (rr_target - rr) * 0.3 * dt
        rr_new = rr + d_rr + rng.normal(0, 2 * dt)
        rr_new = np.clip(rr_new, 8, 40)

        # White blood cell count
        # Leukocytosis with infection
        wbc_target = 8000 + 15000 * infection_severity
        d_wbc = (wbc_target - wbc) * 0.2 * dt
        wbc_new = wbc + d_wbc + rng.normal(0, 500 * dt)
        wbc_new = np.clip(wbc_new, 2000, 30000)

        # Blood pressure (systolic)
        # Hypotension with severe sepsis, improved by fluids/vasopressors
        bp_target = (
            120 - 40 * infection_severity + 20 * fluid_effect + 30 * vasopressor_effect
        )
        d_bp = (bp_target - bp_sys) * 0.5 * dt
        bp_new = bp_sys + d_bp + rng.normal(0, 3 * dt)
        bp_new = np.clip(bp_new, 60, 180)

        # Lactate
        # Elevated with poor perfusion (low BP), cleared with fluids
        lactate_target = 1.0 + 3.0 * infection_severity - 1.0 * fluid_effect
        d_lactate = (lactate_target - lactate) * 0.3 * dt
        lactate_new = lactate + d_lactate + rng.normal(0, 0.1 * dt)
        lactate_new = np.clip(lactate_new, 0.5, 10.0)

        # Build next state (preserve non-modeled features)
        next_state = current_state.copy()
        next_state.update(
            {
                'temperature': float(temp_new),
                'heart_rate': float(hr_new),
                'respiratory_rate': float(rr_new),
                'white_blood_cell_count': float(wbc_new),
                'blood_pressure_sys': float(bp_new),
                'lactate': float(lactate_new),
                # Internal state (not observable)
                '_infection_severity': float(infection_severity),
            }
        )

        return next_state


class RespiratoryDistressModel(DiseaseModel):
    """Acute Respiratory Distress Syndrome (ARDS) progression model.

    Simulated variables:
        - Oxygen saturation (SpO2 %)
        - Respiratory rate (breaths/min)
        - PaO2/FiO2 ratio
        - Lung compliance
        - Heart rate

    Interventions:
        - oxygen_flow: Supplemental oxygen (L/min)
        - peep: Positive end-expiratory pressure (cmH2O)
        - prone_positioning: Improves oxygenation
    """

    def __init__(self, noise_std: float = 0.03):
        self.noise_std = noise_std

    def evolve(
        self,
        current_state: Dict[str, Any],
        dt: float,
        interventions: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Evolve respiratory distress patient state."""
        if rng is None:
            rng = np.random.RandomState()

        # Extract current state
        spo2 = current_state.get('oxygen_saturation', 98.0)
        rr = current_state.get('respiratory_rate', 16)
        pf_ratio = current_state.get('pf_ratio', 400)  # PaO2/FiO2
        hr = current_state.get('heart_rate', 80)

        # Estimate lung injury severity
        lung_injury = np.clip((400 - pf_ratio) / 300.0, 0, 1)

        # Intervention effects
        oxygen_effect = 0.0
        peep_effect = 0.0
        prone_effect = 0.0

        if interventions:
            oxygen_flow = interventions.get('oxygen_flow', 0)
            oxygen_effect = min(oxygen_flow / 15.0, 1.0)  # Max at 15L/min

            peep = interventions.get('peep', 0)
            peep_effect = min(peep / 15.0, 0.5)  # Max benefit at 15 cmH2O

            if interventions.get('prone_positioning', False):
                prone_effect = 0.3

        # SpO2 dynamics
        spo2_target = (
            98 - 12 * lung_injury + 8 * oxygen_effect + 5 * (peep_effect + prone_effect)
        )
        d_spo2 = (spo2_target - spo2) * 0.4 * dt
        spo2_new = spo2 + d_spo2 + rng.normal(0, self.noise_std * 100 * dt)
        spo2_new = np.clip(spo2_new, 70, 100)

        # Respiratory rate (tachypnea with hypoxia)
        rr_target = 14 + 20 * lung_injury - 6 * oxygen_effect
        d_rr = (rr_target - rr) * 0.3 * dt
        rr_new = rr + d_rr + rng.normal(0, 2 * dt)
        rr_new = np.clip(rr_new, 8, 45)

        # PF ratio
        pf_target = (
            400 - 250 * lung_injury + 100 * (oxygen_effect + peep_effect + prone_effect)
        )
        d_pf = (pf_target - pf_ratio) * 0.2 * dt
        pf_new = pf_ratio + d_pf + rng.normal(0, 10 * dt)
        pf_new = np.clip(pf_new, 100, 500)

        # Heart rate (compensatory tachycardia)
        hr_target = 75 + 30 * lung_injury
        d_hr = (hr_target - hr) * 0.3 * dt
        hr_new = hr + d_hr + rng.normal(0, 3 * dt)
        hr_new = np.clip(hr_new, 50, 150)

        # Update state
        next_state = current_state.copy()
        next_state.update(
            {
                'oxygen_saturation': float(spo2_new),
                'respiratory_rate': float(rr_new),
                'pf_ratio': float(pf_new),
                'heart_rate': float(hr_new),
                '_lung_injury': float(lung_injury),
            }
        )

        return next_state


class CardiacEventModel(DiseaseModel):
    """Acute cardiac event (MI/ACS) progression model.

    Simulated variables:
        - Heart rate (bpm)
        - Blood pressure (systolic/diastolic mmHg)
        - Troponin (ng/mL)
        - ST segment elevation (mm)
        - Chest pain score (0-10)

    Interventions:
        - aspirin: Antiplatelet
        - nitrate: Vasodilator
        - beta_blocker: Reduces heart rate and BP
        - pci: Percutaneous coronary intervention
    """

    def __init__(self, noise_std: float = 0.04):
        self.noise_std = noise_std

    def evolve(
        self,
        current_state: Dict[str, Any],
        dt: float,
        interventions: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Evolve cardiac patient state."""
        if rng is None:
            rng = np.random.RandomState()

        # Extract current state
        hr = current_state.get('heart_rate', 75)
        bp_sys = current_state.get('blood_pressure_sys', 130)
        bp_dia = current_state.get('blood_pressure_dia', 80)
        troponin = current_state.get('troponin', 0.01)
        st_elevation = current_state.get('st_elevation', 0.0)
        chest_pain = current_state.get('chest_pain_score', 0)

        # Estimate ischemia severity
        ischemia = np.clip(st_elevation / 3.0 + troponin / 10.0, 0, 1)

        # Intervention effects
        aspirin_effect = 0.0
        nitrate_effect = 0.0
        beta_blocker_effect = 0.0
        pci_effect = 0.0

        if interventions:
            if interventions.get('aspirin', False):
                aspirin_effect = 0.2
            if interventions.get('nitrate', False):
                nitrate_effect = 0.3
            if interventions.get('beta_blocker', False):
                beta_blocker_effect = 0.4
            if interventions.get('pci', False):
                pci_effect = 0.8  # Major improvement

        # Troponin (rises with ongoing ischemia, falls with reperfusion)
        trop_target = 0.01 + 15.0 * ischemia - 10.0 * pci_effect
        d_trop = (trop_target - troponin) * 0.15 * dt
        trop_new = troponin + d_trop + rng.normal(0, 0.5 * dt)
        trop_new = np.clip(trop_new, 0.01, 50.0)

        # ST elevation
        st_target = 0.0 + 3.0 * ischemia - 2.5 * pci_effect
        d_st = (st_target - st_elevation) * 0.3 * dt
        st_new = st_elevation + d_st + rng.normal(0, 0.2 * dt)
        st_new = np.clip(st_new, 0.0, 5.0)

        # Chest pain
        pain_target = 0 + 8 * ischemia - 5 * nitrate_effect - 7 * pci_effect
        d_pain = (pain_target - chest_pain) * 0.5 * dt
        pain_new = chest_pain + d_pain + rng.normal(0, 0.5 * dt)
        pain_new = np.clip(pain_new, 0, 10)

        # Heart rate (reduced by beta blocker)
        hr_target = 75 + 25 * ischemia - 20 * beta_blocker_effect
        d_hr = (hr_target - hr) * 0.4 * dt
        hr_new = hr + d_hr + rng.normal(0, 3 * dt)
        hr_new = np.clip(hr_new, 45, 140)

        # Blood pressure (reduced by nitrate, beta blocker)
        bp_sys_target = (
            130 + 20 * ischemia - 25 * nitrate_effect - 20 * beta_blocker_effect
        )
        d_bp_sys = (bp_sys_target - bp_sys) * 0.3 * dt
        bp_sys_new = bp_sys + d_bp_sys + rng.normal(0, 5 * dt)
        bp_sys_new = np.clip(bp_sys_new, 80, 200)

        bp_dia_target = (
            80 + 10 * ischemia - 15 * nitrate_effect - 10 * beta_blocker_effect
        )
        d_bp_dia = (bp_dia_target - bp_dia) * 0.3 * dt
        bp_dia_new = bp_dia + d_bp_dia + rng.normal(0, 3 * dt)
        bp_dia_new = np.clip(bp_dia_new, 50, 120)

        # Update state
        next_state = current_state.copy()
        next_state.update(
            {
                'heart_rate': float(hr_new),
                'blood_pressure_sys': float(bp_sys_new),
                'blood_pressure_dia': float(bp_dia_new),
                'troponin': float(trop_new),
                'st_elevation': float(st_new),
                'chest_pain_score': float(pain_new),
                '_ischemia_severity': float(ischemia),
            }
        )

        return next_state
