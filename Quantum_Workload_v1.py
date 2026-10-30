# -*- coding: utf-8 -*-

# Quantum_Workload


"""
QUANTUM WORKLOAD LOTO V1

Adaptacija ideje ispitivanja granica slozenosti kvantnog kola.

Sedam izlaznih registara ima po sest kubita.
Registri se simuliraju zasebno.
Nema zajednicke simulacije svih 42 kubita.

Qiskit racuna kompleksne amplitude tokom ucenja.
Svaki registar predstavlja jednu poziciju u rastucoj kombinaciji.

Zajednicka raspodela dobija se uslovljavanjem proizvoda
sedam kvantnih izlaza na:
    1 <= broj_1 < broj_2 < ... < broj_7 <= 39.

Globalna normalizacija i izbor maksimuma racunaju se
dinamickim programiranjem.

Izmedju registara nema kvantnih kapija.
Uslov rastuce kombinacije obradjuje se klasicno.
Unutar svakog registra postoje kvantne kapije sprezanja.

Dubina se bira hronoloskom validacijom.
Odvojeni test ne ucestvuje u izboru dubine.
Zavrsni trening koristi sve istorijske prelaze.

Simulacija koristi tacne amplitude, bez uzorkovanja merenja.
Nema klasifikatora ni klasicnog prediktivnog regresora.

Fajl pravi lokalne NPZ kontrolne tacke kada ga korisnik pokrene.
"""


import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import csv
import hashlib
import math
import sys
import time
import traceback

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import scipy
import qiskit

from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZZGate
from qiskit.quantum_info import Statevector


VERSION = "QUANTUM_WORKLOAD_LOTO_V1_20261030"


FILES = (
    Path("/data/loto7_4710_k86.csv"),
    Path("/data/loto7_4710_k86_loto_2977.csv"),
    Path("/data/loto7_4710_k86_loto_plus_1733.csv"),
)


NUMBERS = 39
PICKS = 7
QUBITS = 6
BASIS_SIZE = 1 << QUBITS

DEPTHS = (1, 2, 3)

RIDGE = 0.01
ANGLE_BOUND = math.pi

MAX_ITERATIONS = 32
MAX_EVALUATIONS = 56
MAX_LINE_SEARCH = 16

GRADIENT_LIMIT = 5.0e-5

VALIDATION_TIE = 1.0e-4

CHECKPOINT_EVERY = 3
PROGRESS_EVERY = 5

MAX_ESTIMATED_MEMORY_GIB = 2.0

SCRIPT_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)

CACHE_DIR = SCRIPT_DIR / "Quantum_Workload_Loto_V1_checkpoints"

BASIS_INDEX = np.arange(BASIS_SIZE, dtype=np.int64)


def say(*values):
    print(*values, flush=True)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def finite(values, label):
    require(
        np.all(np.isfinite(values)),
        f"{label}: rezultat nije konacan.",
    )


def next_power_of_two(number):
    require(number >= 1, "Broj uzoraka mora biti pozitivan.")
    return 1 << (int(number) - 1).bit_length()


def maximum_run(numbers):
    longest = 1
    current = 1

    for previous, value in zip(numbers[:-1], numbers[1:]):
        if value == previous + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    return longest


class OrderedDistribution:
    """
    P(s_1,...,s_K) = product_k A[k,s_k] / Z
    za strogo rastuce indekse.

    A moze biti nenormalizovano po registru.
    Globalno uslovljavanje uklanja tu razliku.
    """

    def __init__(self, numbers, picks):
        self.numbers = int(numbers)
        self.picks = int(picks)

        require(
            1 <= self.picks <= self.numbers,
            "Neispravna velicina kombinacije.",
        )

    def partition(self, log_a, need_marginals=False):
        log_a = np.asarray(log_a, dtype=np.float64)

        require(
            log_a.ndim == 3
            and log_a.shape[1:] == (self.picks, self.numbers),
            "Neispravan oblik kvantnih tezina.",
        )
        finite(log_a, "Logaritmi kvantnih tezina")

        samples = len(log_a)
        k_count = self.picks
        m_count = self.numbers

        prefix = np.full(
            (samples, k_count + 1, m_count + 1),
            -np.inf,
            dtype=np.float64,
        )

        prefix[:, 0, :] = 0.0

        for k in range(1, k_count + 1):
            for j in range(m_count):
                prefix[:, k, j + 1] = np.logaddexp(
                    prefix[:, k, j],
                    prefix[:, k - 1, j] + log_a[:, k - 1, j],
                )

        log_z = prefix[:, k_count, m_count]
        finite(log_z, "Logaritamska normalizacija")

        if not need_marginals:
            return log_z

        suffix = np.full_like(prefix, -np.inf)
        suffix[:, k_count, :] = 0.0

        for k in range(k_count - 1, -1, -1):
            for j in range(m_count - 1, -1, -1):
                suffix[:, k, j] = np.logaddexp(
                    suffix[:, k, j + 1],
                    log_a[:, k, j] + suffix[:, k + 1, j + 1],
                )

        require(
            np.max(np.abs(log_z - suffix[:, 0, 0])) < 2.0e-9,
            "Prednja i zadnja normalizacija se ne slazu.",
        )

        marginal = np.empty_like(log_a)

        for k in range(k_count):
            marginal[:, k, :] = np.exp(
                prefix[:, k, :m_count]
                + log_a[:, k, :]
                + suffix[:, k + 1, 1:]
                - log_z[:, None]
            )

        finite(marginal, "Marginalne verovatnoce")

        require(
            np.min(marginal) >= 0.0,
            "Negativna marginalna verovatnoca.",
        )

        require(
            np.max(np.abs(marginal.sum(axis=2) - 1.0)) < 2.0e-7,
            "Marginalne verovatnoce registra nemaju zbir jedan.",
        )

        inclusion = marginal.sum(axis=1)

        require(
            np.max(inclusion) <= 1.0 + 2.0e-7,
            "Verovatnoca ukljucivanja broja prelazi jedan.",
        )

        require(
            np.max(np.abs(inclusion.sum(axis=1) - k_count))
            < 2.0e-7,
            "Zbir verovatnoca ukljucivanja nije jednak broju izbora.",
        )

        return log_z, marginal

    def log_probabilities(self, log_a, targets):
        targets = np.asarray(targets, dtype=np.int64)

        require(
            targets.shape == (len(log_a), self.picks),
            "Neispravan oblik ciljne kombinacije.",
        )

        require(
            np.all(targets >= 0)
            and np.all(targets < self.numbers)
            and np.all(np.diff(targets, axis=1) > 0),
            "Ciljne kombinacije nisu strogo rastuce.",
        )

        rows = np.arange(len(targets))[:, None]
        ranks = np.arange(self.picks)[None, :]

        selected = log_a[rows, ranks, targets].sum(axis=1)
        result = selected - self.partition(log_a)

        finite(result, "Logaritmi verovatnoca kombinacija")

        require(
            np.max(result) <= 2.0e-8,
            "Verovatnoca kombinacije prelazi jedan.",
        )

        return result

    def maximum(self, log_a):
        log_a = np.asarray(log_a, dtype=np.float64)

        require(
            log_a.shape == (self.picks, self.numbers),
            "Neispravan oblik za trazenje maksimuma.",
        )

        finite(log_a, "Tezine za maksimum")

        dp = np.full(
            (self.picks + 1, self.numbers + 1),
            -np.inf,
            dtype=np.float64,
        )
        take = np.zeros_like(dp, dtype=bool)

        dp[0, :] = 0.0

        for k in range(1, self.picks + 1):
            for j in range(self.numbers):
                skip = dp[k, j]
                choose = dp[k - 1, j] + log_a[k - 1, j]

                if choose > skip:
                    dp[k, j + 1] = choose
                    take[k, j + 1] = True
                else:
                    dp[k, j + 1] = skip

        selected = []
        k = self.picks
        j = self.numbers

        while k > 0:
            require(j > 0, "Neuspesno pracenje putanje maksimuma.")

            if take[k, j]:
                selected.append(j - 1)
                k -= 1

            j -= 1

        selected = np.array(selected[::-1], dtype=np.int64)

        require(
            len(selected) == self.picks
            and np.all(np.diff(selected) > 0),
            "Maksimum ne daje legalnu kombinaciju.",
        )

        log_z = self.partition(log_a[None, :, :])[0]
        log_probability = float(dp[self.picks, self.numbers] - log_z)

        require(
            log_probability <= 2.0e-8,
            "Verovatnoca maksimuma prelazi jedan.",
        )

        require(
            log_probability
            >= -math.log(math.comb(self.numbers, self.picks)) - 2.0e-8,
            "Pronadjena kombinacija nije moguci globalni maksimum.",
        )

        return selected, log_probability

    def independent_log_z(self, log_a):
        """
        Nezavisna pozitivna rekurzija, u prosirenoj preciznosti.
        Koristi se za jednu NEXT raspodelu.
        """

        log_a = np.asarray(log_a, dtype=np.longdouble)
        shifts = np.max(log_a, axis=1)
        weights = np.exp(log_a - shifts[:, None])

        coefficients = np.zeros(self.picks + 1, dtype=np.longdouble)
        coefficients[0] = 1.0

        for j in range(self.numbers):
            for k in range(self.picks - 1, -1, -1):
                coefficients[k + 1] += (
                    coefficients[k] * weights[k, j]
                )

        require(
            coefficients[self.picks] > 0.0,
            "Nezavisna normalizacija nije pozitivna.",
        )

        return float(
            np.log(coefficients[self.picks]) + shifts.sum()
        )


DISTRIBUTION = OrderedDistribution(NUMBERS, PICKS)


def encode_context(context):
    """
    context: broj_uzoraka x 3 x 39.

    Tri prethodna kola kodiraju se u module i faze.
    Sve konstante su fiksni deo kodiranja.
    Nema statistika iz buducih kola.
    """

    context = np.asarray(context, dtype=np.float64)

    require(
        context.ndim == 3 and context.shape[1:] == (3, NUMBERS),
        "Neispravan istorijski kontekst.",
    )

    require(
        np.all((context == 0.0) | (context == 1.0)),
        "Istorijski kontekst nije binaran.",
    )

    require(
        np.all(context.sum(axis=2) == PICKS),
        "Istorijski kontekst ne sadrzi sedam brojeva po kolu.",
    )

    strength = (
        1.0
        + context[:, 0, :]
        + 0.5 * context[:, 1, :]
        + 0.25 * context[:, 2, :]
    )

    phase = math.pi * (
        0.37 * context[:, 0, :]
        - 0.23 * context[:, 1, :]
        + 0.11 * context[:, 2, :]
    )

    encoded = np.zeros(
        (len(context), BASIS_SIZE),
        dtype=np.complex128,
    )

    encoded[:, :NUMBERS] = np.sqrt(strength) * np.exp(1j * phase)

    norms = np.sqrt(np.sum(np.abs(encoded) ** 2, axis=1))
    encoded /= norms[:, None]

    finite(encoded, "Kodirane amplitude")

    require(
        np.max(np.abs(np.sum(np.abs(encoded) ** 2, axis=1) - 1.0))
        < 2.0e-12,
        "Ulazne amplitude nisu normalizovane.",
    )

    return encoded


class QuantumBatch:
    """
    Racunsko grupisanje nezavisnih ulaza.

    Gornji indeksni bitovi sluze samo za vektorizaciju simulatora.
    Kolo deluje iskljucivo na donjih sest kubita.
    Indeksni bitovi nisu dodatni nauceni kvantni registar modela.

    Ovo grupisanje nije tvrdnja o kvantnom ubrzanju.
    """

    def __init__(self, inputs):
        self.inputs = np.asarray(inputs, dtype=np.complex128)

        require(
            self.inputs.ndim == 2
            and self.inputs.shape[1] == BASIS_SIZE
            and len(self.inputs) > 0,
            "Neispravan paket amplituda.",
        )

        finite(self.inputs, "Ulazni paket")

        require(
            np.max(
                np.abs(np.sum(np.abs(self.inputs) ** 2, axis=1) - 1.0)
            ) < 2.0e-10,
            "Paket sadrzi nenormalizovan ulaz.",
        )

        self.samples = len(self.inputs)
        self.padded = next_power_of_two(self.samples)

        packed = np.zeros(
            (self.padded, BASIS_SIZE),
            dtype=np.complex128,
        )

        packed[:self.samples, :] = (
            self.inputs / math.sqrt(self.samples)
        )

        self.packed = packed.reshape(-1)


def gate_for(name, angle):
    if name == "rx":
        return RXGate(float(angle))

    if name == "ry":
        return RYGate(float(angle))

    if name == "rzz":
        return RZZGate(float(angle))

    raise RuntimeError(f"Nepoznata kapija: {name}")


def apply_generator(vector, name, qubits):
    """
    Pauli generator kapije za adjungovani gradijent.

    Ovaj vektor je matematicki medjurezultat gradijenta.
    Ne predstavlja dodatni prediktivni model.
    """

    matrix = np.asarray(vector).reshape(-1, BASIS_SIZE)
    q = qubits[0]

    if name == "rx":
        result = matrix[:, BASIS_INDEX ^ (1 << q)]
        return result.reshape(-1)

    if name == "ry":
        bit = ((BASIS_INDEX >> q) & 1).astype(np.float64)
        signs = 1j * (2.0 * bit - 1.0)

        result = (
            matrix[:, BASIS_INDEX ^ (1 << q)] * signs[None, :]
        )
        return result.reshape(-1)

    if name == "rzz":
        r = qubits[1]

        parity = (
            ((BASIS_INDEX >> q) & 1)
            ^ ((BASIS_INDEX >> r) & 1)
        )

        signs = 1.0 - 2.0 * parity
        return (matrix * signs[None, :]).reshape(-1)

    raise RuntimeError("Nepoznat generator.")


class QuantumModel:
    """
    Jedan nezavisan sestokubitni izlaz za svaki rang kombinacije.
    """

    def __init__(self, depth, heads=PICKS, numbers=NUMBERS):
        self.depth = int(depth)
        self.heads = int(heads)
        self.numbers = int(numbers)

        require(self.depth >= 1, "Dubina mora biti pozitivna.")
        require(
            1 <= self.heads <= self.numbers <= BASIS_SIZE,
            "Neispravna konfiguracija modela.",
        )

        self.parameters_per_head = self.depth * 3 * QUBITS
        self.size = self.heads * self.parameters_per_head

    def seed(self):
        parameters = np.zeros(
            (self.heads, self.depth, 3, QUBITS),
            dtype=np.float64,
        )

        for head in range(self.heads):
            for q in range(QUBITS):
                argument = (head + 1) * (q + 1)

                parameters[head, 0, 0, q] = (
                    0.08 * math.sin(argument)
                )
                parameters[head, 0, 1, q] = (
                    0.10 * math.cos(argument + 0.3)
                )
                parameters[head, 0, 2, q] = (
                    0.05 * math.sin(argument + 0.7)
                )

        return parameters.reshape(-1)

    def operations(self, head_parameters):
        values = np.asarray(head_parameters).reshape(
            self.depth, 3, QUBITS
        )

        operations = []

        for layer in range(self.depth):
            base = layer * 3 * QUBITS

            for q in range(QUBITS):
                operations.append(
                    (
                        "rx",
                        float(values[layer, 0, q]),
                        (q,),
                        base + q,
                    )
                )

            for q in range(QUBITS):
                operations.append(
                    (
                        "rzz",
                        float(values[layer, 1, q]),
                        (q, (q + 1) % QUBITS),
                        base + QUBITS + q,
                    )
                )

            for q in range(QUBITS):
                operations.append(
                    (
                        "ry",
                        float(values[layer, 2, q]),
                        (q,),
                        base + 2 * QUBITS + q,
                    )
                )

        return operations

    def circuit(self, head_parameters):
        circuit = QuantumCircuit(QUBITS)

        for name, angle, qubits, _ in self.operations(head_parameters):
            circuit.append(gate_for(name, angle), list(qubits))

        return circuit

    def forward(self, parameters, batch, keep_states=False):
        parameters = np.asarray(parameters, dtype=np.float64)

        require(
            parameters.shape == (self.size,),
            "Neispravan broj parametara kvantnog kola.",
        )

        finite(parameters, "Parametri kvantnog kola")

        shaped = parameters.reshape(
            self.heads, self.parameters_per_head
        )

        probabilities = np.empty(
            (batch.samples, self.heads, self.numbers),
            dtype=np.float64,
        )

        states = []

        for head in range(self.heads):
            circuit = self.circuit(shaped[head])

            evolved = Statevector(batch.packed).evolve(circuit)
            output = np.asarray(
                evolved.data, dtype=np.complex128
            ).reshape(batch.padded, BASIS_SIZE)

            branch = (
                output[:batch.samples, :]
                * math.sqrt(batch.samples)
            )

            row_norm = np.sum(np.abs(branch) ** 2, axis=1)

            require(
                np.max(np.abs(row_norm - 1.0)) < 2.0e-9,
                "Qiskit evolucija nije sacuvala normu ulaza.",
            )

            if batch.padded > batch.samples:
                require(
                    np.max(np.abs(output[batch.samples:, :]))
                    < 2.0e-12,
                    "Kolo je promenilo prazne indeksne grane.",
                )

            probabilities[:, head, :] = (
                np.abs(branch[:, :self.numbers]) ** 2
            )

            if keep_states:
                states.append(
                    np.array(output.reshape(-1), copy=True)
                )

        finite(probabilities, "Qiskit verovatnoce")

        require(
            np.all(probabilities > 0.0),
            "Kandidat ima nultu kvantnu tezinu; nije prihvacen.",
        )

        if keep_states:
            return probabilities, states

        return probabilities

    def gradient(self, parameters, batch, probabilities, states, residual):
        shaped = np.asarray(parameters).reshape(
            self.heads, self.parameters_per_head
        )

        result = np.zeros_like(shaped)

        for head in range(self.heads):
            state = Statevector(states[head])

            derivative = np.zeros(
                (batch.padded, BASIS_SIZE),
                dtype=np.complex128,
            )

            output = states[head].reshape(
                batch.padded, BASIS_SIZE
            )

            derivative[:batch.samples, :self.numbers] = (
                residual[:, head, :]
                * output[:batch.samples, :self.numbers]
                / probabilities[:, head, :]
            )

            adjoint = Statevector(derivative.reshape(-1))
            operations = self.operations(shaped[head])

            for name, angle, qubits, parameter_index in reversed(
                operations
            ):
                generated = apply_generator(
                    state.data, name, qubits
                )

                result[head, parameter_index] = float(
                    np.real(
                        -1j * np.vdot(adjoint.data, generated)
                    )
                )

                inverse = gate_for(name, -angle)

                state = state.evolve(
                    inverse, qargs=list(qubits)
                )

                adjoint = adjoint.evolve(
                    inverse, qargs=list(qubits)
                )

        result = result.reshape(-1)
        finite(result, "Adjungovani gradijent")

        return result


def objective(parameters, model, batch, targets, distribution):
    probabilities, states = model.forward(
        parameters, batch, keep_states=True
    )

    log_a = np.log(probabilities)

    log_z, marginal = distribution.partition(
        log_a, need_marginals=True
    )

    rows = np.arange(len(targets))[:, None]
    ranks = np.arange(model.heads)[None, :]

    selected = log_a[rows, ranks, targets].sum(axis=1)
    nll = float(np.mean(log_z - selected))

    residual = marginal.copy()
    residual[rows, ranks, targets] -= 1.0

    gradient = model.gradient(
        parameters,
        batch,
        probabilities,
        states,
        residual,
    )

    regularization = 0.5 * RIDGE * float(
        np.mean(np.asarray(parameters) ** 2)
    )

    gradient += RIDGE * parameters / model.size

    loss = nll + regularization

    require(math.isfinite(loss), "Funkcija cilja nije konacna.")
    finite(gradient, "Ukupan gradijent")

    return loss, gradient


def projected_gradient(parameters, gradient):
    projected = np.asarray(gradient, dtype=np.float64).copy()

    lower = parameters <= -ANGLE_BOUND + 1.0e-10
    upper = parameters >= ANGLE_BOUND - 1.0e-10

    projected[lower & (gradient > 0.0)] = 0.0
    projected[upper & (gradient < 0.0)] = 0.0

    return float(np.max(np.abs(projected)))


@dataclass
class Fit:
    model: QuantumModel
    parameters: np.ndarray
    loss: float
    gradient: float
    message: str


def extend_parameters(fit, target_model):
    if fit is None:
        return None

    if (
        fit.model.heads != target_model.heads
        or fit.model.numbers != target_model.numbers
        or fit.model.depth > target_model.depth
    ):
        return None

    result = np.zeros(
        (target_model.heads, target_model.depth, 3, QUBITS),
        dtype=np.float64,
    )

    previous = fit.parameters.reshape(
        fit.model.heads, fit.model.depth, 3, QUBITS
    )

    result[:, :fit.model.depth, :, :] = previous

    return result.reshape(-1)


def checkpoint_path(label, model, inputs, targets, initial):
    digest = hashlib.sha256()

    description = (
        VERSION,
        label,
        model.depth,
        model.heads,
        model.numbers,
        RIDGE,
        ANGLE_BOUND,
        MAX_ITERATIONS,
        MAX_EVALUATIONS,
        MAX_LINE_SEARCH,
        np.__version__,
        scipy.__version__,
        qiskit.__version__,
    )

    digest.update(repr(description).encode("utf-8"))

    for array in (inputs, targets, initial):
        array = np.ascontiguousarray(array)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())

    return CACHE_DIR / (digest.hexdigest() + ".npz")


def save_checkpoint(path, parameters, iterations, done, message):
    temporary = path.with_name(
        path.name + f".{os.getpid()}.tmp"
    )

    try:
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                version=np.array(VERSION),
                parameters=np.asarray(
                    parameters, dtype=np.float64
                ),
                iterations=np.array(iterations, dtype=np.int64),
                done=np.array(bool(done)),
                message=np.array(str(message)),
            )

            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)

    finally:
        if temporary.exists():
            temporary.unlink()


class EvaluationBudgetReached(Exception):
    pass


def fit_model(inputs, targets, depth, label, warm=None):
    model = QuantumModel(depth)
    batch = QuantumBatch(inputs)

    initial = extend_parameters(warm, model)

    if initial is None:
        initial = model.seed()

    path = checkpoint_path(
        label, model, inputs, targets, initial
    )

    parameters = initial.copy()
    iterations = 0
    done = False
    message = "Pocetak."

    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as saved:
                require(
                    str(saved["version"].item()) == VERSION,
                    "Verzija kontrolne tacke nije odgovarajuca.",
                )

                candidate = np.asarray(
                    saved["parameters"], dtype=np.float64
                )

                require(
                    candidate.shape == (model.size,),
                    "Neispravan oblik sacuvanih parametara.",
                )

                finite(candidate, "Sacuvani parametri")

                require(
                    np.max(np.abs(candidate))
                    <= ANGLE_BOUND + 1.0e-10,
                    "Sacuvani parametri su izvan granica.",
                )

                parameters = candidate.copy()
                iterations = int(saved["iterations"].item())
                done = bool(saved["done"].item())
                message = str(saved["message"].item())

                require(
                    0 <= iterations <= MAX_ITERATIONS,
                    "Neispravan sacuvani broj iteracija.",
                )

            say(label, "| ucitana kontrolna tacka.")

        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            EOFError,
            BadZipFile,
        ) as error:
            say(
                label,
                "| kontrolna tacka nije prihvacena:",
                str(error),
            )

            parameters = initial.copy()
            iterations = 0
            done = False
            message = "Ponovljen pocetak."

    start = time.perf_counter()

    loss, gradient = objective(
        parameters, model, batch, targets, DISTRIBUTION
    )

    best_parameters = parameters.copy()
    best_loss = float(loss)
    best_gradient = gradient.copy()

    calls = 0

    say()
    say(label)
    say(
        f"Slojevi: {depth} | parametri: {model.size} "
        f"| uzorci: {len(targets)}"
    )

    def function(candidate):
        nonlocal calls
        nonlocal best_parameters, best_loss, best_gradient

        if calls >= MAX_EVALUATIONS:
            raise EvaluationBudgetReached()

        calls += 1

        value, derivative = objective(
            candidate, model, batch, targets, DISTRIBUTION
        )

        if value < best_loss:
            best_loss = float(value)
            best_parameters = np.array(candidate, copy=True)
            best_gradient = np.array(derivative, copy=True)

        if calls == 1 or calls % PROGRESS_EVERY == 0:
            elapsed = time.perf_counter() - start

            say(
                f"Evaluacija {calls}/{MAX_EVALUATIONS} "
                f"| cilj {best_loss:.9f} "
                f"| gradijent "
                f"{projected_gradient(best_parameters, best_gradient):.3e} "
                f"| proteklo {elapsed:.1f} s"
            )

        return value, derivative

    def callback(_candidate):
        nonlocal iterations

        iterations += 1

        if iterations % CHECKPOINT_EVERY == 0:
            save_checkpoint(
                path,
                best_parameters,
                iterations,
                False,
                "Optimizacija u toku.",
            )

    try:
        if not done and iterations < MAX_ITERATIONS:
            remaining = MAX_ITERATIONS - iterations

            optimization = minimize(
                function,
                best_parameters,
                method="L-BFGS-B",
                jac=True,
                bounds=[
                    (-ANGLE_BOUND, ANGLE_BOUND)
                ] * model.size,
                callback=callback,
                options={
                    "maxiter": remaining,
                    "maxfun": MAX_EVALUATIONS,
                    "maxls": MAX_LINE_SEARCH,
                    "maxcor": 10,
                    "ftol": 1.0e-11,
                    "gtol": 1.0e-6,
                },
            )

            message = str(optimization.message)

        elif not done:
            message = "Dostignut budzet iteracija."

    except EvaluationBudgetReached:
        message = "Dostignut budzet evaluacija."

    except KeyboardInterrupt:
        save_checkpoint(
            path,
            best_parameters,
            iterations,
            False,
            "Korisnicki prekid.",
        )
        raise

    except (
        FloatingPointError,
        ValueError,
        RuntimeError,
        np.linalg.LinAlgError,
    ) as error:
        message = (
            "Numericki kandidat nije prihvacen. "
            "Zadrzan je najbolji prethodni kandidat. "
            + str(error)
        )

    checked_loss, checked_gradient = objective(
        best_parameters, model, batch, targets, DISTRIBUTION
    )

    require(
        abs(checked_loss - best_loss)
        < 2.0e-8 * (1.0 + abs(best_loss)),
        "Ponovljeni prora cun cilja se ne slaze.",
    )

    gradient_norm = projected_gradient(
        best_parameters, checked_gradient
    )

    save_checkpoint(
        path,
        best_parameters,
        min(iterations, MAX_ITERATIONS),
        True,
        message,
    )

    say(f"Najbolji cilj: {checked_loss:.9f}")
    say(f"Max|projektovani gradijent|: {gradient_norm:.3e}")
    say("Optimizator:", message)

    if gradient_norm > GRADIENT_LIMIT:
        say(
            "DIJAGNOSTIKA: stacionarnost nije potvrdjena; "
            "koristi se najbolji numericki vazeci kandidat."
        )

    return Fit(
        model=model,
        parameters=best_parameters,
        loss=float(checked_loss),
        gradient=gradient_norm,
        message=message,
    )


def predict_log_a(fit, inputs):
    batch = QuantumBatch(inputs)

    probabilities = fit.model.forward(
        fit.parameters, batch
    )

    return np.log(probabilities)


def evaluate(fit, inputs, targets, with_hits=False):
    log_a = predict_log_a(fit, inputs)

    nll = -float(
        np.mean(
            DISTRIBUTION.log_probabilities(log_a, targets)
        )
    )

    if not with_hits:
        return nll

    histogram = np.zeros(PICKS + 1, dtype=np.int64)

    for index in range(len(targets)):
        selected, _ = DISTRIBUTION.maximum(log_a[index])
        hits = len(set(selected).intersection(targets[index]))
        histogram[hits] += 1

    return nll, histogram


def load_draws(path):
    raw = path.read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()

    text = raw.decode("utf-8-sig")
    rows = []
    blanks = 0

    for line_number, row in enumerate(
        csv.reader(text.splitlines()), start=1
    ):
        if not row or all(not value.strip() for value in row):
            blanks += 1
            continue

        require(
            len(row) == PICKS,
            f"{path.name}, red {line_number}: "
            "ocekivano je sedam kolona.",
        )

        try:
            values = [int(value.strip()) for value in row]
        except ValueError as error:
            raise RuntimeError(
                f"{path.name}, red {line_number}: "
                "sve vrednosti moraju biti celi brojevi."
            ) from error

        require(
            len(set(values)) == PICKS
            and min(values) >= 1
            and max(values) <= NUMBERS,
            f"{path.name}, red {line_number}: "
            "neispravna kombinacija.",
        )

        rows.append(sorted(value - 1 for value in values))

    require(
        len(rows) >= 100,
        f"{path.name}: potrebno je najmanje 100 kola.",
    )

    draws = np.asarray(rows, dtype=np.int64)

    occupancy = np.zeros(
        (len(draws), NUMBERS), dtype=np.float64
    )

    occupancy[
        np.arange(len(draws))[:, None], draws
    ] = 1.0

    say()
    say("ULAZ:", path.name)
    say(f"Redova: {len(draws)} | praznih redova: {blanks}")
    say("SHA256:", fingerprint)
    say("Redosled kola ostaje redosled iz CSV-a.")

    return draws, occupancy, fingerprint


def dataset(draws, occupancy):
    indices = np.arange(len(draws))

    context = np.stack(
        [
            occupancy[np.maximum(indices - lag, 0)]
            for lag in range(3)
        ],
        axis=1,
    )

    encoded = encode_context(context)

    return encoded[:-1], draws[1:], encoded[-1:]


def resource_profile(sample_count, depth):
    model = QuantumModel(depth)
    shaped = model.seed().reshape(
        model.heads, model.parameters_per_head
    )

    circuit = model.circuit(shaped[0])

    padded = next_power_of_two(sample_count)

    state_bytes = padded * BASIS_SIZE * 16
    input_bytes = sample_count * BASIS_SIZE * 16

    dp_bytes = (
        sample_count
        * (PICKS + 1)
        * (NUMBERS + 1)
        * 8
    )

    # Konzervativna procena radnih nizova.
    # Nije merenje stvarnog vrha memorije procesa.
    estimated_bytes = (
        256 * 1024**2
        + (PICKS + 12) * state_bytes
        + 5 * input_bytes
        + 6 * dp_bytes
    )

    return {
        "depth": depth,
        "parameters": model.size,
        "gates_per_register": len(circuit.data),
        "total_gates": model.heads * len(circuit.data),
        "circuit_depth": circuit.depth(),
        "batch_index_bits": padded.bit_length() - 1,
        "estimated_gib": estimated_bytes / 1024**3,
    }


def check_resources(loaded):
    say()
    say("PROCENA RESURSA PRE TRENINGA")
    say(
        "Broj kapija i dubina odnose se na nauceni deo kola; "
        "priprema ulaznih amplituda nije ukljucena."
    )
    say(
        "Memorija je konzervativna procena za simulator, "
        "a ne izmereni vrh procesa."
    )

    for path, draws, _, _ in loaded:
        samples = len(draws) - 1

        say()
        say(path.name)

        for depth in DEPTHS:
            profile = resource_profile(samples, depth)

            say(
                f"Slojevi {depth} "
                f"| parametri {profile['parameters']} "
                f"| kapije/registar "
                f"{profile['gates_per_register']} "
                f"| kapije ukupno {profile['total_gates']} "
                f"| dubina/registar "
                f"{profile['circuit_depth']} "
                f"| procena RAM "
                f"{profile['estimated_gib']:.3f} GiB"
            )

            require(
                profile["estimated_gib"]
                <= MAX_ESTIMATED_MEMORY_GIB,
                f"{path.name}: procenjena memorija prelazi "
                f"{MAX_ESTIMATED_MEMORY_GIB:.2f} GiB. "
                "Trening nije zapocet.",
            )


def self_checks():
    say()
    say("Pokrecem Qiskit i matematicke samoprovere.")

    toy = OrderedDistribution(6, 3)

    weights = (
        0.15
        + np.arange(18, dtype=np.float64).reshape(3, 6) / 40.0
    )

    log_a = np.log(weights)
    log_z, marginal = toy.partition(
        log_a[None, :, :], need_marginals=True
    )

    all_sets = list(combinations(range(6), 3))

    products = np.array(
        [
            np.prod(
                weights[np.arange(3), np.asarray(selected)]
            )
            for selected in all_sets
        ],
        dtype=np.float64,
    )

    exact_z = float(products.sum())
    exact_probabilities = products / exact_z

    exact_marginal = np.zeros((3, 6), dtype=np.float64)

    for selected, probability in zip(
        all_sets, exact_probabilities
    ):
        exact_marginal[
            np.arange(3), np.asarray(selected)
        ] += probability

    require(
        abs(log_z[0] - math.log(exact_z)) < 2.0e-12,
        "Normalizacija ne prolazi potpuno nabrajanje.",
    )

    require(
        np.max(np.abs(marginal[0] - exact_marginal))
        < 2.0e-12,
        "Marginalne verovatnoce nisu ispravne.",
    )

    selected, log_probability = toy.maximum(log_a)

    require(
        tuple(selected)
        == all_sets[int(np.argmax(exact_probabilities))],
        "Globalni maksimum nije ispravan.",
    )

    require(
        abs(
            math.exp(log_probability)
            - float(np.max(exact_probabilities))
        ) < 2.0e-12,
        "Verovatnoca maksimuma nije ispravna.",
    )

    require(
        abs(toy.independent_log_z(log_a) - log_z[0])
        < 2.0e-12,
        "Nezavisna normalizacija nije ispravna.",
    )

    for picks in (1, 6):
        boundary = OrderedDistribution(6, picks)
        neutral = np.zeros((1, picks, 6), dtype=np.float64)

        value = boundary.partition(neutral)[0]

        require(
            abs(value - math.log(math.comb(6, picks)))
            < 2.0e-12,
            "Granicni slucaj raspodele nije ispravan.",
        )

    neutral_full = np.zeros(
        (1, PICKS, NUMBERS), dtype=np.float64
    )

    require(
        abs(
            DISTRIBUTION.partition(neutral_full)[0]
            - math.log(math.comb(NUMBERS, PICKS))
        ) < 2.0e-11,
        "Provera broja svih kombinacija nije ispravna.",
    )

    inputs = np.zeros((3, BASIS_SIZE), dtype=np.complex128)

    for row in range(3):
        for j in range(6):
            inputs[row, j] = (
                0.8 + 0.1 * math.sin((row + 1) * (j + 1))
            ) * np.exp(0.13j * (row + 1) * (j + 1))

    inputs /= np.sqrt(
        np.sum(np.abs(inputs) ** 2, axis=1)
    )[:, None]

    targets = np.array(
        [[0, 2], [1, 5], [2, 4]],
        dtype=np.int64,
    )

    batch = QuantumBatch(inputs)
    small_distribution = OrderedDistribution(6, 2)

    maximum_gradient_error = 0.0
    maximum_batch_error = 0.0

    for depth in (1, 2):
        model = QuantumModel(depth, heads=2, numbers=6)

        parameters = (
            0.13
            * np.sin(np.arange(model.size, dtype=np.float64) + 0.4)
        )

        value, analytic = objective(
            parameters,
            model,
            batch,
            targets,
            small_distribution,
        )

        require(math.isfinite(value), "Samoprovera cilja nije konacna.")

        epsilon = 2.0e-6

        for index in range(model.size):
            plus = parameters.copy()
            minus = parameters.copy()

            plus[index] += epsilon
            minus[index] -= epsilon

            value_plus = objective(
                plus,
                model,
                batch,
                targets,
                small_distribution,
            )[0]

            value_minus = objective(
                minus,
                model,
                batch,
                targets,
                small_distribution,
            )[0]

            numerical = (value_plus - value_minus) / (2.0 * epsilon)

            error = abs(numerical - analytic[index])
            maximum_gradient_error = max(
                maximum_gradient_error, error
            )

            require(
                error < 2.0e-5 * (1.0 + abs(analytic[index])),
                "Adjungovani gradijent ne prolazi numericku proveru.",
            )

        _, states = model.forward(
            parameters, batch, keep_states=True
        )

        shaped = parameters.reshape(
            model.heads, model.parameters_per_head
        )

        for head in range(model.heads):
            circuit = model.circuit(shaped[head])

            packed = states[head].reshape(
                batch.padded, BASIS_SIZE
            )

            for row in range(len(inputs)):
                direct = Statevector(inputs[row]).evolve(circuit)

                reconstructed = (
                    packed[row] * math.sqrt(batch.samples)
                )

                error = float(
                    np.max(
                        np.abs(direct.data - reconstructed)
                    )
                )

                maximum_batch_error = max(
                    maximum_batch_error, error
                )

                require(
                    error < 2.0e-11,
                    "Paketna simulacija nije jednaka pojedinacnoj.",
                )

    # Dodatni sloj sa nultim uglovima mora biti identitet.
    shallow = QuantumModel(1, heads=2, numbers=6)
    deep = QuantumModel(2, heads=2, numbers=6)

    shallow_parameters = shallow.seed()

    shallow_fit = Fit(
        shallow,
        shallow_parameters,
        0.0,
        0.0,
        "Samoprovera.",
    )

    extended = extend_parameters(shallow_fit, deep)

    q_shallow = shallow.forward(shallow_parameters, batch)
    q_deep = deep.forward(extended, batch)

    require(
        np.max(np.abs(q_shallow - q_deep)) < 2.0e-12,
        "Prosirenje nultim slojem nije sacuvalo model.",
    )

    # Kolo mora reagovati na naucene parametre.
    changed = shallow_parameters.copy()
    changed[-1] += 0.19

    changed_q = shallow.forward(changed, batch)

    require(
        np.max(np.abs(changed_q - q_shallow)) > 1.0e-7,
        "Parametri ne menjaju kvantni izlaz.",
    )

    say("Samoprovere su prosle.")
    say(
        f"Najveca greska gradijenta: "
        f"{maximum_gradient_error:.3e}"
    )
    say(
        f"Najveca greska paketne Qiskit simulacije: "
        f"{maximum_batch_error:.3e}"
    )


def direct_next_check(fit, next_input, expected_log_a):
    shaped = fit.parameters.reshape(
        fit.model.heads, fit.model.parameters_per_head
    )

    maximum_error = 0.0

    for head in range(fit.model.heads):
        circuit = fit.model.circuit(shaped[head])

        direct = Statevector(next_input[0]).evolve(circuit)
        probabilities = (
            np.abs(direct.data[:NUMBERS]) ** 2
        )

        expected = np.exp(expected_log_a[head])

        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(probabilities - expected))),
        )

    require(
        maximum_error < 2.0e-10,
        "Pojedinacna Qiskit provera NEXT izlaza nije prosla.",
    )

    return maximum_error


def process_file(path, draws, occupancy, fingerprint):
    inputs, targets, next_input = dataset(draws, occupancy)

    count = len(targets)

    cut60 = int(0.60 * count)
    cut70 = int(0.70 * count)
    cut80 = int(0.80 * count)

    folds = (
        (cut60, cut70),
        (cut70, cut80),
    )

    previous_by_fold = [None, None]
    candidates = []

    say()
    say("=" * 70)
    say("OBRADA:", path.name)
    say(f"Istorijskih prelaza: {count}")
    say(
        "Validacija: 60%-70% i 70%-80%; "
        "odvojeni test: poslednjih 20% prelaza."
    )

    for depth in DEPTHS:
        total_nll = 0.0
        total_rows = 0
        fold_fits = []

        for fold_index, (train_end, valid_end) in enumerate(folds):
            label = (
                f"{path.name} | slojevi {depth} "
                f"| validacija {fold_index + 1}"
            )

            fit = fit_model(
                inputs[:train_end],
                targets[:train_end],
                depth,
                label,
                warm=previous_by_fold[fold_index],
            )

            score = evaluate(
                fit,
                inputs[train_end:valid_end],
                targets[train_end:valid_end],
            )

            rows = valid_end - train_end

            total_nll += score * rows
            total_rows += rows

            previous_by_fold[fold_index] = fit
            fold_fits.append(fit)

            say(f"Validacioni NLL: {score:.9f}")

        validation_nll = total_nll / total_rows

        candidates.append(
            {
                "depth": depth,
                "validation": validation_nll,
                "fits": fold_fits,
            }
        )

        say()
        say(
            f"Slojevi {depth} | objedinjeni validacioni NLL: "
            f"{validation_nll:.9f}"
        )

    minimum_score = min(
        candidate["validation"] for candidate in candidates
    )

    equivalent = [
        candidate
        for candidate in candidates
        if candidate["validation"]
        <= minimum_score + VALIDATION_TIE
    ]

    selected = min(
        equivalent, key=lambda candidate: candidate["depth"]
    )

    selected_depth = selected["depth"]

    say()
    say("IZBOR DUBINE")

    for candidate in candidates:
        say(
            f"Slojevi {candidate['depth']} "
            f"| validacioni NLL {candidate['validation']:.9f}"
        )

    say(f"Izabrano slojeva: {selected_depth}")
    say(
        f"Tolerancija za prednost jednostavnijeg kola: "
        f"{VALIDATION_TIE:.6f} NLL"
    )

    test_fit = fit_model(
        inputs[:cut80],
        targets[:cut80],
        selected_depth,
        f"{path.name} | odvojeni test",
        warm=selected["fits"][-1],
    )

    test_nll, histogram = evaluate(
        test_fit,
        inputs[cut80:],
        targets[cut80:],
        with_hits=True,
    )

    say()
    say(f"Odvojeni test NLL: {test_nll:.9f}")
    say(
        "Ovaj rezultat ne menja vec izabranu dubinu."
    )

    final_fit = fit_model(
        inputs,
        targets,
        selected_depth,
        f"{path.name} | zavrsni trening cele istorije",
        warm=test_fit,
    )

    next_log_a = predict_log_a(
        final_fit, next_input
    )[0]

    log_z, marginal = DISTRIBUTION.partition(
        next_log_a[None, :, :],
        need_marginals=True,
    )

    selected_numbers, log_probability = (
        DISTRIBUTION.maximum(next_log_a)
    )

    alternative_log_z = (
        DISTRIBUTION.independent_log_z(next_log_a)
    )

    normalization_error = abs(
        alternative_log_z - float(log_z[0])
    )

    require(
        normalization_error < 2.0e-9,
        "Nezavisna NEXT normalizacija nije prosla.",
    )

    normalization_ratio = math.exp(
        alternative_log_z - float(log_z[0])
    )

    qiskit_error = direct_next_check(
        final_fit, next_input, next_log_a
    )

    inclusion = marginal[0].sum(axis=0)

    # Poslednjih osam konteksta sluzi samo za dijagnostiku.
    # Zavrsni trening je vec koristio sve istorijske prelaze.
    recent_contexts = np.vstack(
        (inputs[-7:], next_input)
    )

    recent_log_a = predict_log_a(
        final_fit, recent_contexts
    )

    _, recent_marginal = DISTRIBUTION.partition(
        recent_log_a, need_marginals=True
    )

    recent_inclusion = recent_marginal.sum(axis=1)

    sensitivity = float(
        np.max(np.ptp(recent_inclusion, axis=0))
    )

    numbers = tuple(
        int(value + 1) for value in selected_numbers
    )

    warnings = []

    if maximum_run(numbers) >= 5:
        warnings.append(
            "NEXT sadrzi najmanje pet uzastopnih brojeva."
        )

    if numbers[-1] - numbers[0] <= 9:
        warnings.append(
            "Svih sedam brojeva nalazi se u rasponu od deset brojeva."
        )

    if any(
        np.array_equal(selected_numbers, historical)
        for historical in draws
    ):
        warnings.append(
            "NEXT ponavlja kombinaciju prisutnu u istoriji."
        )

    if np.array_equal(selected_numbers, draws[-1]):
        warnings.append(
            "NEXT je jednak poslednjem istorijskom kolu."
        )

    if sensitivity < 1.0e-8:
        warnings.append(
            "Izlaz slabo reaguje na proverene istorijske kontekste."
        )

    if final_fit.gradient > GRADIENT_LIMIT:
        warnings.append(
            "Zavrsni optimizacioni stacionarni uslov nije potvrdjen."
        )

    if test_fit.gradient > GRADIENT_LIMIT:
        warnings.append(
            "Test model je dobijen uz nepotvrdjenu stacionarnost."
        )

    profile = resource_profile(
        len(targets), selected_depth
    )

    say()
    say("REZULTAT:", path.name)
    say("NEXT:", " ".join(map(str, numbers)))
    say(f"Izabrano slojeva: {selected_depth}")
    say(
        f"Validacioni NLL izabranog modela: "
        f"{selected['validation']:.9f}"
    )
    say(f"Odvojeni test NLL: {test_nll:.9f}")
    say(
        f"Modelna verovatnoca NEXT: "
        f"{math.exp(log_probability):.12g}"
    )
    say(
        f"Zbir verovatnoca prema nezavisnoj rekurziji: "
        f"{normalization_ratio:.12f}"
    )
    say(
        f"Greska logaritamske normalizacije: "
        f"{normalization_error:.3e}"
    )
    say(
        f"Zbir verovatnoca ukljucivanja: "
        f"{inclusion.sum():.12f}"
    )
    say(
        f"Broj kombinacija obuhvacenih raspodelom: "
        f"{math.comb(NUMBERS, PICKS):,}"
    )
    say(
        "Maksimum je odredjen dinamickim programiranjem; "
        "kombinacije nisu pojedinacno generisane."
    )
    say(f"Greska pojedinacne Qiskit provere: {qiskit_error:.3e}")
    say(f"Dijagnosticka osetljivost: {sensitivity:.9e}")
    say(
        f"Zavrsni projektovani gradijent: "
        f"{final_fit.gradient:.3e}"
    )
    say(
        f"Parametri: {profile['parameters']} "
        f"| ukupno naucenih kapija: "
        f"{profile['total_gates']}"
    )

    say("Test histogram broja pogodaka:")

    for hits, frequency in enumerate(histogram):
        say(f"{hits}: {int(frequency)}")

    mean_hits = float(
        np.dot(np.arange(PICKS + 1), histogram)
        / histogram.sum()
    )

    say(f"Prosecan broj pogodaka na testu: {mean_hits:.6f}")

    say("Verovatnoce ukljucivanja u NEXT raspodeli:")

    for value, probability in enumerate(inclusion, start=1):
        say(f"{value:2d}: {probability:.9f}")

    for warning in warnings:
        say("DIJAGNOSTIKA:", warning)

    if warnings:
        say(
            "Rezultat nije menjan radi uklanjanja "
            "dijagnostickog obrasca."
        )

    return {
        "name": path.name,
        "next": numbers,
        "depth": selected_depth,
        "test_nll": test_nll,
        "inclusion": inclusion,
        "sensitivity": sensitivity,
        "last": tuple(int(value + 1) for value in draws[-1]),
        "fingerprint": fingerprint,
    }


def main():
    say("QUANTUM WORKLOAD LOTO V1")
    say("Python:", sys.version.split()[0])
    say("Qiskit:", qiskit.__version__)
    say("NumPy:", np.__version__)
    say("SciPy:", scipy.__version__)
    say()
    say("Lokalna Qiskit simulacija kompleksnih amplituda.")
    say("Sedam registara od po sest kubita.")
    say("Registri se simuliraju zasebno.")
    say("Unutar registara: RX, RZZ i RY kapije.")
    say("Izmedju registara nema kvantnih kapija.")
    say(
        "Globalno uslovljavanje na sedam rastucih brojeva "
        "racuna se klasicno."
    )
    say(
        "Qiskit evolucija koristi se tokom treninga, "
        "validacije, testa i NEXT proracuna."
    )
    say(
        "Gradijent i optimizator su klasicni; "
        "nema klasicnog prediktivnog regresora."
    )
    say("Nema nasumicnog uzorkovanja ni generisanja kombinacija.")
    say("Porede se dubine:", ", ".join(map(str, DEPTHS)))
    say(
        f"Budzet po treningu: najvise {MAX_ITERATIONS} iteracije "
        f"i {MAX_EVALUATIONS} optimizacionih evaluacija."
    )
    say(
        "Pocetna i zavrsna numericka provera funkcije cilja "
        "rade se dodatno."
    )
    say(
        "Budzet ogranicava racunski rad; "
        "konvergencija i globalni optimum nisu garantovani."
    )
    say(
        "Test je hronoloski, jedan korak unapred: "
        "prethodno stvarno kolo moze biti ulaz narednog test uzorka."
    )
    say(
        "Kontrolne tacke cuvaju najbolji vazeci kandidat, "
        "ukljucujuci dostignut optimizacioni budzet."
    )

    loaded = []

    for path in FILES:
        draws, occupancy, fingerprint = load_draws(path)
        loaded.append(
            (path, draws, occupancy, fingerprint)
        )

    # Provera svih ulaza i procena resursa prethode treningu.
    check_resources(loaded)
    self_checks()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    say()
    say("Kontrolne tacke:", str(CACHE_DIR))

    results = []

    for path, draws, occupancy, fingerprint in loaded:
        result = process_file(
            path, draws, occupancy, fingerprint
        )
        results.append(result)

    say()
    say("=" * 70)
    say("ZAVRSNI NEXT REZULTATI")

    for result in results:
        say(
            f"{result['name']} | NEXT: "
            + " ".join(map(str, result["next"]))
            + f" | slojevi: {result['depth']}"
            + f" | test NLL: {result['test_nll']:.9f}"
        )

    for first, second in combinations(results, 2):
        if first["next"] == second["next"]:
            difference = float(
                np.max(
                    np.abs(
                        first["inclusion"]
                        - second["inclusion"]
                    )
                )
            )

            say()
            say(
                "DIJAGNOSTIKA: isti NEXT za",
                first["name"],
                "i",
                second["name"],
            )

            say(
                f"Najveca razlika izlaznih verovatnoca: "
                f"{difference:.9e}"
            )

            say(
                f"Osetljivosti na istoriju: "
                f"{first['sensitivity']:.9e} i "
                f"{second['sensitivity']:.9e}"
            )

            say(
                "Isto poslednje kolo:",
                first["last"] == second["last"],
            )

            say(
                "Rezultat nije menjan radi vestackog razlikovanja."
            )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        say()
        say("PREKINUTO NA ZAHTEV KORISNIKA.")
        say(
            "Tokom treninga najbolji kandidat cuva se "
            "u kontrolnoj tacki."
        )
        sys.exit(130)

    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        say()
        say("ZAUSTAVLJENO:", str(error))
        say(
            "Zavrsni skup od tri NEXT rezultata nije potvrdjen."
        )
        traceback.print_exc()
        sys.exit(1)



"""
QUANTUM WORKLOAD LOTO V1
Python: 3.11.13
Qiskit: 1.4.4
NumPy: 1.26.4
SciPy: 1.17.1

Lokalna Qiskit simulacija kompleksnih amplituda.
Sedam registara od po sest kubita.
Registri se simuliraju zasebno.
Unutar registara: RX, RZZ i RY kapije.
Izmedju registara nema kvantnih kapija.
Globalno uslovljavanje na sedam rastucih brojeva racuna se klasicno.
Qiskit evolucija koristi se tokom treninga, validacije, testa i NEXT proracuna.
Gradijent i optimizator su klasicni; nema klasicnog prediktivnog regresora.
Nema nasumicnog uzorkovanja ni generisanja kombinacija.
Porede se dubine: 1, 2, 3
Budzet po treningu: najvise 32 iteracije i 56 optimizacionih evaluacija.
Pocetna i zavrsna numericka provera funkcije cilja rade se dodatno.
Budzet ogranicava racunski rad; konvergencija i globalni optimum nisu garantovani.
Test je hronoloski, jedan korak unapred: prethodno stvarno kolo moze biti ulaz narednog test uzorka.
Kontrolne tacke cuvaju najbolji vazeci kandidat, ukljucujuci dostignut optimizacioni budzet.

ULAZ: loto7_4710_k86.csv
Redova: 4710 | praznih redova: 0
SHA256: e51a5ab684401459535996f50fd0b0540b0af6d2b86bca49825cf381d759c85d
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_2977.csv
Redova: 2977 | praznih redova: 0
SHA256: 69ae55edbff87383f9bb4a25df4ba6b44acbf23404db6e7fa3642de0442479f4
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_plus_1733.csv
Redova: 1733 | praznih redova: 0
SHA256: 9d708eb88434b55a38bbc69474a869955e0295f1bed401cf10dc177d7d8fef39
Redosled kola ostaje redosled iz CSV-a.

PROCENA RESURSA PRE TRENINGA
Broj kapija i dubina odnose se na nauceni deo kola; priprema ulaznih amplituda nije ukljucena.
Memorija je konzervativna procena za simulator, a ne izmereni vrh procesa.

loto7_4710_k86.csv
Slojevi 1 | parametri 126 | kapije/registar 18 | kapije ukupno 126 | dubina/registar 8 | procena RAM 0.488 GiB
Slojevi 2 | parametri 252 | kapije/registar 36 | kapije ukupno 252 | dubina/registar 16 | procena RAM 0.488GiB
Slojevi 3 | parametri 378 | kapije/registar 54 | kapije ukupno 378 | dubina/registar 24 | procena RAM 0.488GiB

loto7_4710_k86_loto_2977.csv
Slojevi 1 | parametri 126 | kapije/registar 18 | kapije ukupno 126 | dubina/registar 8 | procena RAM 0.381 GiB
Slojevi 2 | parametri 252 | kapije/registar 36 | kapije ukupno 252 | dubina/registar 16 | procena RAM 0.381GiB
Slojevi 3 | parametri 378 | kapije/registar 54 | kapije ukupno 378 | dubina/registar 24 | procena RAM 0.381GiB

loto7_4710_k86_loto_plus_1733.csv
Slojevi 1 | parametri 126 | kapije/registar 18 | kapije ukupno 126 | dubina/registar 8 | procena RAM 0.320 GiB
Slojevi 2 | parametri 252 | kapije/registar 36 | kapije ukupno 252 | dubina/registar 16 | procena RAM 0.320GiB
Slojevi 3 | parametri 378 | kapije/registar 54 | kapije ukupno 378 | dubina/registar 24 | procena RAM 0.320GiB

Pokrecem Qiskit i matematicke samoprovere.
Samoprovere su prosle.
Najveca greska gradijenta: 3.629e-10
Najveca greska paketne Qiskit simulacije: 3.395e-16

Kontrolne tacke: /Quantum_Workload/Quantum_Workload_Loto_V1_checkpoints

======================================================================
OBRADA: loto7_4710_k86.csv
Istorijskih prelaza: 4709
Validacija: 60%-70% i 70%-80%; odvojeni test: poslednjih 20% prelaza.

loto7_4710_k86.csv | slojevi 1 | validacija 1
Slojevi: 1 | parametri: 126 | uzorci: 2825
Evaluacija 1/56 | cilj 16.854733407 | gradijent 9.346e-02 | proteklo 1.1 s
Evaluacija 5/56 | cilj 16.722640495 | gradijent 1.814e-02 | proteklo 3.4 s
Evaluacija 10/56 | cilj 16.712949681 | gradijent 3.098e-03 | proteklo 6.1 s
Evaluacija 15/56 | cilj 16.711465031 | gradijent 2.523e-03 | proteklo 8.9 s
Evaluacija 20/56 | cilj 16.710343039 | gradijent 4.516e-03 | proteklo 11.6 s
Evaluacija 25/56 | cilj 16.708813475 | gradijent 2.289e-03 | proteklo 14.3 s
Evaluacija 30/56 | cilj 16.707486172 | gradijent 5.939e-03 | proteklo 17.0 s
Najbolji cilj: 16.706364072
Max|projektovani gradijent|: 3.002e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.
Validacioni NLL: 16.749161750

loto7_4710_k86.csv | slojevi 1 | validacija 2
Slojevi: 1 | parametri: 126 | uzorci: 3296
Evaluacija 1/56 | cilj 16.856311780 | gradijent 8.741e-02 | proteklo 1.1 s
Evaluacija 5/56 | cilj 16.724279681 | gradijent 1.876e-02 | proteklo 3.5 s
Evaluacija 10/56 | cilj 16.714788113 | gradijent 2.510e-03 | proteklo 6.3 s
Evaluacija 15/56 | cilj 16.713473938 | gradijent 2.427e-03 | proteklo 9.1 s
Evaluacija 20/56 | cilj 16.712713936 | gradijent 4.755e-03 | proteklo 11.8 s
Evaluacija 25/56 | cilj 16.711253424 | gradijent 1.890e-03 | proteklo 14.6 s
Evaluacija 30/56 | cilj 16.709643769 | gradijent 3.414e-03 | proteklo 17.4 s
Najbolji cilj: 16.708355206
Max|projektovani gradijent|: 4.264e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.
Validacioni NLL: 16.762097392

Slojevi 1 | objedinjeni validacioni NLL: 16.755629571

loto7_4710_k86.csv | slojevi 2 | validacija 1
Slojevi: 2 | parametri: 252 | uzorci: 2825
Evaluacija 1/56 | cilj 16.706239242 | gradijent 1.614e-02 | proteklo 2.1 s
Evaluacija 5/56 | cilj 16.702948156 | gradijent 8.171e-03 | proteklo 6.3 s
Evaluacija 10/56 | cilj 16.699561050 | gradijent 5.562e-03 | proteklo 11.5 s
Evaluacija 15/56 | cilj 16.695934910 | gradijent 4.893e-03 | proteklo 16.8 s
Evaluacija 20/56 | cilj 16.691457241 | gradijent 6.285e-03 | proteklo 22.0 s
Evaluacija 25/56 | cilj 16.687512059 | gradijent 4.847e-03 | proteklo 27.3 s
Evaluacija 30/56 | cilj 16.684969907 | gradijent 3.465e-03 | proteklo 32.7 s
Najbolji cilj: 16.684045925
Max|projektovani gradijent|: 2.398e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.
Validacioni NLL: 16.745086620

loto7_4710_k86.csv | slojevi 2 | validacija 2
Slojevi: 2 | parametri: 252 | uzorci: 3296
Evaluacija 1/56 | cilj 16.708224788 | gradijent 2.058e-02 | proteklo 2.1 s
Evaluacija 5/56 | cilj 16.703487129 | gradijent 9.135e-03 | proteklo 6.3 s
Evaluacija 10/56 | cilj 16.698508088 | gradijent 5.959e-03 | proteklo 11.6 s
Evaluacija 15/56 | cilj 16.695004368 | gradijent 5.037e-03 | proteklo 16.8 s
Evaluacija 20/56 | cilj 16.693125603 | gradijent 5.080e-03 | proteklo 22.0 s
Evaluacija 25/56 | cilj 16.689909906 | gradijent 4.144e-03 | proteklo 27.3 s
Evaluacija 30/56 | cilj 16.687352753 | gradijent 5.757e-03 | proteklo 32.5 s
Evaluacija 35/56 | cilj 16.685674555 | gradijent 3.691e-03 | proteklo 37.7 s
Najbolji cilj: 16.685674555
Max|projektovani gradijent|: 3.691e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.
Validacioni NLL: 16.748276732

Slojevi 2 | objedinjeni validacioni NLL: 16.746681676

loto7_4710_k86.csv | slojevi 3 | validacija 1
Slojevi: 3 | parametri: 378 | uzorci: 2825
Evaluacija 1/56 | cilj 16.683965409 | gradijent 2.403e-03 | proteklo 3.1 s
Evaluacija 5/56 | cilj 16.683338213 | gradijent 3.224e-03 | proteklo 9.3 s
Evaluacija 10/56 | cilj 16.681568708 | gradijent 5.933e-03 | proteklo 17.0 s
Evaluacija 15/56 | cilj 16.678892743 | gradijent 4.780e-03 | proteklo 24.9 s
Evaluacija 20/56 | cilj 16.676608235 | gradijent 6.134e-03 | proteklo 32.6 s
Evaluacija 25/56 | cilj 16.674218944 | gradijent 7.533e-03 | proteklo 40.4 s
Evaluacija 30/56 | cilj 16.671808060 | gradijent 9.557e-03 | proteklo 48.1 s
Najbolji cilj: 16.669729156
Max|projektovani gradijent|: 1.035e-02
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.
Validacioni NLL: 16.756916913

loto7_4710_k86.csv | slojevi 3 | validacija 2
Slojevi: 3 | parametri: 378 | uzorci: 3296
Evaluacija 1/56 | cilj 16.685589291 | gradijent 3.690e-03 | proteklo 3.1 s
Evaluacija 5/56 | cilj 16.685101347 | gradijent 4.010e-03 | proteklo 9.2 s
Evaluacija 10/56 | cilj 16.683956578 | gradijent 2.705e-03 | proteklo 16.9 s
Evaluacija 15/56 | cilj 16.681505554 | gradijent 8.108e-03 | proteklo 24.6 s
Evaluacija 20/56 | cilj 16.678094331 | gradijent 1.381e-02 | proteklo 32.3 s
Evaluacija 25/56 | cilj 16.676128132 | gradijent 4.704e-03 | proteklo 40.0 s
Evaluacija 30/56 | cilj 16.673256841 | gradijent 9.399e-03 | proteklo 47.7 s
Evaluacija 35/56 | cilj 16.669884984 | gradijent 7.686e-03 | proteklo 55.4 s
Najbolji cilj: 16.669884984
Max|projektovani gradijent|: 7.686e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.
Validacioni NLL: 16.731185768

Slojevi 3 | objedinjeni validacioni NLL: 16.744051341

IZBOR DUBINE
Slojevi 1 | validacioni NLL 16.755629571
Slojevi 2 | validacioni NLL 16.746681676
Slojevi 3 | validacioni NLL 16.744051341
Izabrano slojeva: 3
Tolerancija za prednost jednostavnijeg kola: 0.000100 NLL

loto7_4710_k86.csv | odvojeni test
Slojevi: 3 | parametri: 378 | uzorci: 3767
Evaluacija 1/56 | cilj 16.677584491 | gradijent 1.587e-02 | proteklo 3.1 s
Evaluacija 5/56 | cilj 16.674211147 | gradijent 6.719e-03 | proteklo 9.3 s
Evaluacija 10/56 | cilj 16.672199632 | gradijent 7.582e-03 | proteklo 17.2 s
Evaluacija 15/56 | cilj 16.670458626 | gradijent 8.670e-03 | proteklo 25.1 s
Evaluacija 20/56 | cilj 16.668274396 | gradijent 8.267e-03 | proteklo 32.8 s
Evaluacija 25/56 | cilj 16.665344792 | gradijent 6.407e-03 | proteklo 40.6 s
Evaluacija 30/56 | cilj 16.662140763 | gradijent 1.196e-02 | proteklo 48.4 s
Najbolji cilj: 16.660310401
Max|projektovani gradijent|: 1.822e-02
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.

Odvojeni test NLL: 16.727114748
Ovaj rezultat ne menja vec izabranu dubinu.

loto7_4710_k86.csv | zavrsni trening cele istorije
Slojevi: 3 | parametri: 378 | uzorci: 4709
Evaluacija 1/56 | cilj 16.673757269 | gradijent 3.020e-02 | proteklo 6.4 s
Evaluacija 5/56 | cilj 16.669604604 | gradijent 6.879e-03 | proteklo 19.2 s
Evaluacija 10/56 | cilj 16.666374481 | gradijent 7.618e-03 | proteklo 35.3 s
Evaluacija 15/56 | cilj 16.664045381 | gradijent 4.939e-03 | proteklo 51.3 s
Evaluacija 20/56 | cilj 16.662117076 | gradijent 5.077e-03 | proteklo 67.3 s
Evaluacija 25/56 | cilj 16.659333870 | gradijent 4.797e-03 | proteklo 83.3 s
Evaluacija 30/56 | cilj 16.657355593 | gradijent 9.087e-03 | proteklo 99.3 s
Evaluacija 35/56 | cilj 16.654965835 | gradijent 4.195e-03 | proteklo 115.3 s
Najbolji cilj: 16.654551790
Max|projektovani gradijent|: 3.970e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.

REZULTAT: loto7_4710_k86.csv
NEXT: 18 x 32 y 36 z 39
Izabrano slojeva: 3
Validacioni NLL izabranog modela: 16.744051341
Odvojeni test NLL: 16.727114748
Modelna verovatnoca NEXT: 1.27689531769e-06
Zbir verovatnoca prema nezavisnoj rekurziji: 1.000000000000
Greska logaritamske normalizacije: 1.776e-15
Zbir verovatnoca ukljucivanja: 7.000000000000
Broj kombinacija obuhvacenih raspodelom: 15,380,937
Maksimum je odredjen dinamickim programiranjem; kombinacije nisu pojedinacno generisane.
Greska pojedinacne Qiskit provere: 6.939e-18
Dijagnosticka osetljivost: 1.512050712e-01
Zavrsni projektovani gradijent: 3.970e-03
Parametri: 378 | ukupno naucenih kapija: 378
Test histogram broja pogodaka:
0: 200
1: 396
2: 246
3: 88
4: 12
5: 0
6: 0
7: 0
Prosecan broj pogodaka na testu: 1.273885
Verovatnoce ukljucivanja u NEXT raspodeli:
 1: 0.158309399
 2: 0.157834126
 3: 0.178912305
 4: 0.161590399
 5: 0.158756104
 6: 0.248469890
 7: 0.164829964
 8: 0.160373793
 9: 0.246796754
10: 0.197321181
11: 0.156123516
12: 0.167061890
13: 0.175311000
14: 0.148681584
15: 0.223786989
16: 0.199025900
17: 0.151794946
18: 0.163724061
19: 0.145270995
20: 0.141250198
21: 0.161936649
22: 0.235686755
23: 0.172372276
24: 0.175499153
25: 0.157532623
26: 0.199594035
27: 0.166948513
28: 0.170666498
29: 0.186971554
30: 0.153662017
31: 0.155542737
32: 0.276509998
33: 0.186303787
34: 0.183019280
35: 0.174484553
36: 0.187646634
37: 0.184321666
38: 0.175147015
39: 0.190929262
DIJAGNOSTIKA: Zavrsni optimizacioni stacionarni uslov nije potvrdjen.
DIJAGNOSTIKA: Test model je dobijen uz nepotvrdjenu stacionarnost.
Rezultat nije menjan radi uklanjanja dijagnostickog obrasca.

======================================================================
OBRADA: loto7_4710_k86_loto_2977.csv
Istorijskih prelaza: 2976
Validacija: 60%-70% i 70%-80%; odvojeni test: poslednjih 20% prelaza.

loto7_4710_k86_loto_2977.csv | slojevi 1 | validacija 1
Slojevi: 1 | parametri: 126 | uzorci: 1785
Evaluacija 1/56 | cilj 16.859783995 | gradijent 1.159e-01 | proteklo 0.5 s
Evaluacija 5/56 | cilj 16.716712270 | gradijent 1.977e-02 | proteklo 1.6 s
Evaluacija 10/56 | cilj 16.704818596 | gradijent 3.461e-03 | proteklo 3.0 s
Evaluacija 15/56 | cilj 16.703453331 | gradijent 2.512e-03 | proteklo 4.3 s
Evaluacija 20/56 | cilj 16.702172919 | gradijent 3.854e-03 | proteklo 5.7 s
Evaluacija 25/56 | cilj 16.700038344 | gradijent 3.338e-03 | proteklo 7.1 s
Evaluacija 30/56 | cilj 16.696948275 | gradijent 3.894e-03 | proteklo 8.5 s
Najbolji cilj: 16.694611322
Max|projektovani gradijent|: 4.481e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.
Validacioni NLL: 16.754213861

loto7_4710_k86_loto_2977.csv | slojevi 1 | validacija 2
Slojevi: 1 | parametri: 126 | uzorci: 2083
Evaluacija 1/56 | cilj 16.855786227 | gradijent 1.100e-01 | proteklo 1.1 s
Evaluacija 5/56 | cilj 16.717367137 | gradijent 1.839e-02 | proteklo 3.2 s
Evaluacija 10/56 | cilj 16.706440414 | gradijent 4.505e-03 | proteklo 6.0 s
Evaluacija 15/56 | cilj 16.704742209 | gradijent 3.333e-03 | proteklo 8.8 s
Evaluacija 20/56 | cilj 16.703560080 | gradijent 3.049e-03 | proteklo 11.5 s
Evaluacija 25/56 | cilj 16.701340613 | gradijent 3.276e-03 | proteklo 14.2 s
Evaluacija 30/56 | cilj 16.698970945 | gradijent 4.027e-03 | proteklo 16.8 s
Najbolji cilj: 16.698033636
Max|projektovani gradijent|: 5.843e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.
Validacioni NLL: 16.744009199

Slojevi 1 | objedinjeni validacioni NLL: 16.749120105

loto7_4710_k86_loto_2977.csv | slojevi 2 | validacija 1
Slojevi: 2 | parametri: 252 | uzorci: 1785
Evaluacija 1/56 | cilj 16.694442896 | gradijent 1.474e-02 | proteklo 1.1 s
Evaluacija 5/56 | cilj 16.688451440 | gradijent 1.115e-02 | proteklo 3.1 s
Evaluacija 10/56 | cilj 16.681268097 | gradijent 9.119e-03 | proteklo 5.8 s
Evaluacija 15/56 | cilj 16.672868680 | gradijent 8.731e-03 | proteklo 8.3 s
Evaluacija 20/56 | cilj 16.664045984 | gradijent 2.034e-02 | proteklo 10.9 s
Evaluacija 25/56 | cilj 16.658604425 | gradijent 7.219e-03 | proteklo 13.5 s
Evaluacija 30/56 | cilj 16.652387061 | gradijent 7.501e-03 | proteklo 16.1 s
Najbolji cilj: 16.647306462
Max|projektovani gradijent|: 9.604e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.
Validacioni NLL: 16.774083636

loto7_4710_k86_loto_2977.csv | slojevi 2 | validacija 2
Slojevi: 2 | parametri: 252 | uzorci: 2083
Evaluacija 1/56 | cilj 16.697916601 | gradijent 1.472e-02 | proteklo 2.2 s
Evaluacija 5/56 | cilj 16.693533414 | gradijent 1.746e-02 | proteklo 6.4 s
Evaluacija 10/56 | cilj 16.686452792 | gradijent 8.247e-03 | proteklo 11.6 s
Evaluacija 15/56 | cilj 16.681111150 | gradijent 9.400e-03 | proteklo 16.7 s
Evaluacija 20/56 | cilj 16.676868878 | gradijent 3.554e-03 | proteklo 22.0 s
Evaluacija 25/56 | cilj 16.673285749 | gradijent 1.384e-02 | proteklo 27.2 s
Evaluacija 30/56 | cilj 16.668739447 | gradijent 7.329e-03 | proteklo 32.5 s
Najbolji cilj: 16.663520623
Max|projektovani gradijent|: 7.247e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.
Validacioni NLL: 16.769370622

Slojevi 2 | objedinjeni validacioni NLL: 16.771731090

loto7_4710_k86_loto_2977.csv | slojevi 3 | validacija 1
Slojevi: 3 | parametri: 378 | uzorci: 1785
Evaluacija 1/56 | cilj 16.647137038 | gradijent 9.809e-03 | proteklo 1.5 s
Evaluacija 5/56 | cilj 16.644942489 | gradijent 5.637e-03 | proteklo 4.6 s
Evaluacija 10/56 | cilj 16.639725384 | gradijent 7.142e-03 | proteklo 8.4 s
Evaluacija 15/56 | cilj 16.634901570 | gradijent 5.303e-03 | proteklo 12.2 s
Evaluacija 20/56 | cilj 16.631428102 | gradijent 5.569e-03 | proteklo 16.0 s
Evaluacija 25/56 | cilj 16.626390706 | gradijent 8.831e-03 | proteklo 19.8 s
Evaluacija 30/56 | cilj 16.620528382 | gradijent 6.453e-03 | proteklo 23.6 s
Evaluacija 35/56 | cilj 16.616578425 | gradijent 7.439e-03 | proteklo 27.3 s
Najbolji cilj: 16.616578425
Max|projektovani gradijent|: 7.439e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.
Validacioni NLL: 16.777086654

loto7_4710_k86_loto_2977.csv | slojevi 3 | validacija 2
Slojevi: 3 | parametri: 378 | uzorci: 2083
Evaluacija 1/56 | cilj 16.663405337 | gradijent 8.626e-03 | proteklo 3.1 s
Evaluacija 5/56 | cilj 16.660495742 | gradijent 1.029e-02 | proteklo 9.2 s
Evaluacija 10/56 | cilj 16.656633351 | gradijent 8.035e-03 | proteklo 16.9 s
Evaluacija 15/56 | cilj 16.651854888 | gradijent 7.359e-03 | proteklo 24.8 s
Evaluacija 20/56 | cilj 16.646514352 | gradijent 5.018e-03 | proteklo 32.4 s
Evaluacija 25/56 | cilj 16.642134585 | gradijent 3.808e-03 | proteklo 40.2 s
Evaluacija 30/56 | cilj 16.637097271 | gradijent 7.371e-03 | proteklo 47.8 s
Najbolji cilj: 16.632773111
Max|projektovani gradijent|: 6.351e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.
Validacioni NLL: 16.780432487

Slojevi 3 | objedinjeni validacioni NLL: 16.778756759

IZBOR DUBINE
Slojevi 1 | validacioni NLL 16.749120105
Slojevi 2 | validacioni NLL 16.771731090
Slojevi 3 | validacioni NLL 16.778756759
Izabrano slojeva: 1
Tolerancija za prednost jednostavnijeg kola: 0.000100 NLL

loto7_4710_k86_loto_2977.csv | odvojeni test
Slojevi: 1 | parametri: 126 | uzorci: 2380
Evaluacija 1/56 | cilj 16.703800132 | gradijent 1.999e-02 | proteklo 1.1 s
Evaluacija 5/56 | cilj 16.700769181 | gradijent 5.387e-03 | proteklo 3.2 s
Evaluacija 10/56 | cilj 16.699817499 | gradijent 2.251e-03 | proteklo 5.9 s
Evaluacija 15/56 | cilj 16.699114715 | gradijent 3.280e-03 | proteklo 8.6 s
Evaluacija 20/56 | cilj 16.698706090 | gradijent 9.274e-04 | proteklo 11.3 s
Evaluacija 25/56 | cilj 16.698360660 | gradijent 1.319e-03 | proteklo 14.0 s
Evaluacija 30/56 | cilj 16.697691371 | gradijent 2.277e-03 | proteklo 16.7 s
Najbolji cilj: 16.697157677
Max|projektovani gradijent|: 4.178e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.

Odvojeni test NLL: 16.736549686
Ovaj rezultat ne menja vec izabranu dubinu.

loto7_4710_k86_loto_2977.csv | zavrsni trening cele istorije
Slojevi: 1 | parametri: 126 | uzorci: 2976
Evaluacija 1/56 | cilj 16.705139711 | gradijent 1.438e-02 | proteklo 1.1 s
Evaluacija 5/56 | cilj 16.701542854 | gradijent 4.393e-03 | proteklo 3.3 s
Evaluacija 10/56 | cilj 16.700749671 | gradijent 4.530e-03 | proteklo 6.1 s
Evaluacija 15/56 | cilj 16.699629967 | gradijent 4.661e-03 | proteklo 8.8 s
Evaluacija 20/56 | cilj 16.697862175 | gradijent 3.425e-03 | proteklo 11.5 s
Evaluacija 25/56 | cilj 16.696071502 | gradijent 3.164e-03 | proteklo 14.2 s
Evaluacija 30/56 | cilj 16.694260122 | gradijent 3.566e-03 | proteklo 17.0 s
Evaluacija 35/56 | cilj 16.693036657 | gradijent 2.735e-03 | proteklo 19.7 s
Najbolji cilj: 16.693036657
Max|projektovani gradijent|: 2.735e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.

REZULTAT: loto7_4710_k86_loto_2977.csv
NEXT: 3 x 22 y 35 z 39
Izabrano slojeva: 1
Validacioni NLL izabranog modela: 16.749120105
Odvojeni test NLL: 16.736549686
Modelna verovatnoca NEXT: 8.81540445588e-07
Zbir verovatnoca prema nezavisnoj rekurziji: 1.000000000000
Greska logaritamske normalizacije: 3.553e-15
Zbir verovatnoca ukljucivanja: 7.000000000000
Broj kombinacija obuhvacenih raspodelom: 15,380,937
Maksimum je odredjen dinamickim programiranjem; kombinacije nisu pojedinacno generisane.
Greska pojedinacne Qiskit provere: 6.939e-18
Dijagnosticka osetljivost: 1.585982996e-01
Zavrsni projektovani gradijent: 2.735e-03
Parametri: 126 | ukupno naucenih kapija: 126
Test histogram broja pogodaka:
0: 128
1: 250
2: 163
3: 47
4: 8
5: 0
6: 0
7: 0
Prosecan broj pogodaka na testu: 1.256711
Verovatnoce ukljucivanja u NEXT raspodeli:
 1: 0.190971787
 2: 0.133520234
 3: 0.258238017
 4: 0.142448013
 5: 0.150055218
 6: 0.225241685
 7: 0.145044740
 8: 0.160006015
 9: 0.206412231
10: 0.155347083
11: 0.170253493
12: 0.149768135
13: 0.184141230
14: 0.167174819
15: 0.224727664
16: 0.177514979
17: 0.152594939
18: 0.238709282
19: 0.167855799
20: 0.137445353
21: 0.188969006
22: 0.239596849
23: 0.160926083
24: 0.184182294
25: 0.169527874
26: 0.131875175
27: 0.156816142
28: 0.156843744
29: 0.184463856
30: 0.193573921
31: 0.190514378
32: 0.180020373
33: 0.149368463
34: 0.211983678
35: 0.169277822
36: 0.221702294
37: 0.149316310
38: 0.194605687
39: 0.228965333
DIJAGNOSTIKA: Zavrsni optimizacioni stacionarni uslov nije potvrdjen.
DIJAGNOSTIKA: Test model je dobijen uz nepotvrdjenu stacionarnost.
Rezultat nije menjan radi uklanjanja dijagnostickog obrasca.

======================================================================
OBRADA: loto7_4710_k86_loto_plus_1733.csv
Istorijskih prelaza: 1732
Validacija: 60%-70% i 70%-80%; odvojeni test: poslednjih 20% prelaza.

loto7_4710_k86_loto_plus_1733.csv | slojevi 1 | validacija 1
Slojevi: 1 | parametri: 126 | uzorci: 1039
Evaluacija 1/56 | cilj 16.830339981 | gradijent 1.091e-01 | proteklo 0.5 s
Evaluacija 5/56 | cilj 16.685481500 | gradijent 2.007e-02 | proteklo 1.6 s
Evaluacija 10/56 | cilj 16.671972253 | gradijent 7.013e-03 | proteklo 2.9 s
Evaluacija 15/56 | cilj 16.667088373 | gradijent 3.307e-03 | proteklo 4.2 s
Evaluacija 20/56 | cilj 16.663057513 | gradijent 5.753e-03 | proteklo 5.5 s
Evaluacija 25/56 | cilj 16.659004799 | gradijent 1.727e-02 | proteklo 6.9 s
Evaluacija 30/56 | cilj 16.654881785 | gradijent 7.034e-03 | proteklo 8.2 s
Evaluacija 35/56 | cilj 16.650504160 | gradijent 4.496e-03 | proteklo 9.5 s
Najbolji cilj: 16.650504160
Max|projektovani gradijent|: 4.496e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.
Validacioni NLL: 16.794236382

loto7_4710_k86_loto_plus_1733.csv | slojevi 1 | validacija 2
Slojevi: 1 | parametri: 126 | uzorci: 1212
Evaluacija 1/56 | cilj 16.835283701 | gradijent 1.011e-01 | proteklo 0.5 s
Evaluacija 5/56 | cilj 16.696173507 | gradijent 2.019e-02 | proteklo 1.6 s
Evaluacija 10/56 | cilj 16.683629500 | gradijent 6.884e-03 | proteklo 3.0 s
Evaluacija 15/56 | cilj 16.679788244 | gradijent 3.894e-03 | proteklo 4.3 s
Evaluacija 20/56 | cilj 16.676884685 | gradijent 5.364e-03 | proteklo 5.7 s
Evaluacija 25/56 | cilj 16.670568330 | gradijent 2.381e-02 | proteklo 7.2 s
Evaluacija 30/56 | cilj 16.666189286 | gradijent 6.852e-03 | proteklo 8.5 s
Najbolji cilj: 16.663050727
Max|projektovani gradijent|: 5.909e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.
Validacioni NLL: 16.792416888

Slojevi 1 | objedinjeni validacioni NLL: 16.793326635

loto7_4710_k86_loto_plus_1733.csv | slojevi 2 | validacija 1
Slojevi: 2 | parametri: 252 | uzorci: 1039
Evaluacija 1/56 | cilj 16.650289827 | gradijent 4.400e-02 | proteklo 1.0 s
Evaluacija 5/56 | cilj 16.642158826 | gradijent 1.279e-02 | proteklo 3.1 s
Evaluacija 10/56 | cilj 16.630542599 | gradijent 9.767e-03 | proteklo 5.6 s
Evaluacija 15/56 | cilj 16.621024953 | gradijent 8.154e-03 | proteklo 8.2 s
Evaluacija 20/56 | cilj 16.616974881 | gradijent 5.708e-03 | proteklo 10.7 s
Evaluacija 25/56 | cilj 16.612110971 | gradijent 8.383e-03 | proteklo 13.3 s
Evaluacija 30/56 | cilj 16.605035116 | gradijent 2.288e-02 | proteklo 15.9 s
Najbolji cilj: 16.598053431
Max|projektovani gradijent|: 8.301e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.
Validacioni NLL: 16.847329928

loto7_4710_k86_loto_plus_1733.csv | slojevi 2 | validacija 2
Slojevi: 2 | parametri: 252 | uzorci: 1212
Evaluacija 1/56 | cilj 16.662800567 | gradijent 2.439e-02 | proteklo 1.0 s
Evaluacija 5/56 | cilj 16.655334930 | gradijent 1.336e-02 | proteklo 3.0 s
Evaluacija 10/56 | cilj 16.647207482 | gradijent 7.454e-03 | proteklo 5.6 s
Evaluacija 15/56 | cilj 16.637514495 | gradijent 1.171e-02 | proteklo 8.1 s
Evaluacija 20/56 | cilj 16.634045357 | gradijent 7.984e-03 | proteklo 10.7 s
Evaluacija 25/56 | cilj 16.631369944 | gradijent 4.551e-03 | proteklo 13.2 s
Evaluacija 30/56 | cilj 16.627941666 | gradijent 8.291e-03 | proteklo 15.8 s
Najbolji cilj: 16.625270673
Max|projektovani gradijent|: 1.021e-02
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.
Validacioni NLL: 16.804429520

Slojevi 2 | objedinjeni validacioni NLL: 16.825879724

loto7_4710_k86_loto_plus_1733.csv | slojevi 3 | validacija 1
Slojevi: 3 | parametri: 378 | uzorci: 1039
Evaluacija 1/56 | cilj 16.597916905 | gradijent 9.025e-03 | proteklo 1.5 s
Evaluacija 5/56 | cilj 16.595067063 | gradijent 7.049e-03 | proteklo 4.5 s
Evaluacija 10/56 | cilj 16.588477164 | gradijent 1.272e-02 | proteklo 8.2 s
Evaluacija 15/56 | cilj 16.581609146 | gradijent 8.235e-03 | proteklo 12.0 s
Evaluacija 20/56 | cilj 16.574857346 | gradijent 1.595e-02 | proteklo 15.7 s
Evaluacija 25/56 | cilj 16.568075726 | gradijent 1.809e-02 | proteklo 19.4 s
Evaluacija 30/56 | cilj 16.561656695 | gradijent 1.066e-02 | proteklo 23.1 s
Najbolji cilj: 16.556893390
Max|projektovani gradijent|: 6.202e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.
Validacioni NLL: 16.895513284

loto7_4710_k86_loto_plus_1733.csv | slojevi 3 | validacija 2
Slojevi: 3 | parametri: 378 | uzorci: 1212
Evaluacija 1/56 | cilj 16.625127643 | gradijent 1.021e-02 | proteklo 1.5 s
Evaluacija 5/56 | cilj 16.624039216 | gradijent 4.558e-03 | proteklo 4.5 s
Evaluacija 10/56 | cilj 16.620979555 | gradijent 7.447e-03 | proteklo 8.3 s
Evaluacija 15/56 | cilj 16.617795853 | gradijent 7.126e-03 | proteklo 12.0 s
Evaluacija 20/56 | cilj 16.613974873 | gradijent 4.508e-03 | proteklo 15.8 s
Evaluacija 25/56 | cilj 16.610913143 | gradijent 6.949e-03 | proteklo 19.5 s
Evaluacija 30/56 | cilj 16.605544829 | gradijent 6.717e-03 | proteklo 23.3 s
Najbolji cilj: 16.601907512
Max|projektovani gradijent|: 1.233e-02
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.
Validacioni NLL: 16.813534935

Slojevi 3 | objedinjeni validacioni NLL: 16.854524109

IZBOR DUBINE
Slojevi 1 | validacioni NLL 16.793326635
Slojevi 2 | validacioni NLL 16.825879724
Slojevi 3 | validacioni NLL 16.854524109
Izabrano slojeva: 1
Tolerancija za prednost jednostavnijeg kola: 0.000100 NLL

loto7_4710_k86_loto_plus_1733.csv | odvojeni test
Slojevi: 1 | parametri: 126 | uzorci: 1385
Evaluacija 1/56 | cilj 16.679272316 | gradijent 2.416e-02 | proteklo 0.5 s
Evaluacija 5/56 | cilj 16.672780093 | gradijent 4.449e-03 | proteklo 1.6 s
Evaluacija 10/56 | cilj 16.671181034 | gradijent 4.536e-03 | proteklo 2.9 s
Evaluacija 15/56 | cilj 16.669745852 | gradijent 2.285e-03 | proteklo 4.3 s
Evaluacija 20/56 | cilj 16.669181776 | gradijent 1.804e-03 | proteklo 5.6 s
Evaluacija 25/56 | cilj 16.668268452 | gradijent 1.676e-03 | proteklo 7.0 s
Evaluacija 30/56 | cilj 16.667425341 | gradijent 2.513e-03 | proteklo 8.3 s
Najbolji cilj: 16.667144881
Max|projektovani gradijent|: 1.984e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.

Odvojeni test NLL: 16.698736241
Ovaj rezultat ne menja vec izabranu dubinu.

loto7_4710_k86_loto_plus_1733.csv | zavrsni trening cele istorije
Slojevi: 1 | parametri: 126 | uzorci: 1732
Evaluacija 1/56 | cilj 16.673615267 | gradijent 1.846e-02 | proteklo 0.6 s
Evaluacija 5/56 | cilj 16.667509896 | gradijent 5.739e-03 | proteklo 1.7 s
Evaluacija 10/56 | cilj 16.665733297 | gradijent 2.844e-03 | proteklo 3.1 s
Evaluacija 15/56 | cilj 16.664930424 | gradijent 1.609e-03 | proteklo 4.5 s
Evaluacija 20/56 | cilj 16.664398192 | gradijent 5.599e-03 | proteklo 5.9 s
Evaluacija 25/56 | cilj 16.663782858 | gradijent 3.732e-03 | proteklo 7.4 s
Evaluacija 30/56 | cilj 16.663084177 | gradijent 3.439e-03 | proteklo 8.8 s
Najbolji cilj: 16.662534836
Max|projektovani gradijent|: 6.026e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji numericki vazeci kandidat.

REZULTAT: loto7_4710_k86_loto_plus_1733.csv
NEXT: 3 x 14 y 26 z 33
Izabrano slojeva: 1
Validacioni NLL izabranog modela: 16.793326635
Odvojeni test NLL: 16.698736241
Modelna verovatnoca NEXT: 9.15727951071e-07
Zbir verovatnoca prema nezavisnoj rekurziji: 1.000000000000
Greska logaritamske normalizacije: 0.000e+00
Zbir verovatnoca ukljucivanja: 7.000000000000
Broj kombinacija obuhvacenih raspodelom: 15,380,937
Maksimum je odredjen dinamickim programiranjem; kombinacije nisu pojedinacno generisane.
Greska pojedinacne Qiskit provere: 6.939e-18
Dijagnosticka osetljivost: 1.810532326e-01
Zavrsni projektovani gradijent: 6.026e-03
Parametri: 126 | ukupno naucenih kapija: 126
Test histogram broja pogodaka:
0: 82
1: 140
2: 96
3: 27
4: 2
5: 0
6: 0
7: 0
Prosecan broj pogodaka na testu: 1.213256
Verovatnoce ukljucivanja u NEXT raspodeli:
 1: 0.173377515
 2: 0.172884151
 3: 0.205880004
 4: 0.181397189
 5: 0.146408538
 6: 0.154184122
 7: 0.166379028
 8: 0.136624460
 9: 0.170044894
10: 0.214336955
11: 0.171868111
12: 0.133847763
13: 0.200821968
14: 0.225698732
15: 0.154178913
16: 0.231528815
17: 0.147205409
18: 0.167236283
19: 0.148232557
20: 0.170321563
21: 0.148175102
22: 0.165631672
23: 0.147977656
24: 0.154074243
25: 0.148817728
26: 0.267659805
27: 0.233851148
28: 0.169846101
29: 0.150158668
30: 0.167111261
31: 0.151342414
32: 0.269882346
33: 0.242845537
34: 0.192730621
35: 0.152966536
36: 0.202995417
37: 0.192350383
38: 0.166406176
39: 0.202720218
DIJAGNOSTIKA: Zavrsni optimizacioni stacionarni uslov nije potvrdjen.
DIJAGNOSTIKA: Test model je dobijen uz nepotvrdjenu stacionarnost.
Rezultat nije menjan radi uklanjanja dijagnostickog obrasca.

======================================================================
ZAVRSNI NEXT REZULTATI
loto7_4710_k86.csv | NEXT: 18 x 32 y 36 z 39 | slojevi: 3 | test NLL: 16.727114748
loto7_4710_k86_loto_2977.csv | NEXT: 3 x 22 y 35 z 39 | slojevi: 1 | test NLL: 16.736549686
loto7_4710_k86_loto_plus_1733.csv | NEXT: 3 x 14 y 26 z 33 | slojevi: 1 | test NLL: 16.698736241
"""



"""
Analiza granica složenosti kvantnog kola: može pomoći da biram složenost modela prema validacionom rezultatu i trošku izvršavanja. 

Modelu sistematski menjam složenost kvantnog kola i proverimo gde dodatna složenost prestaje da pomaže.

- Ista istorija kodira se u kvantne amplitude.
- Porede se kola sa različitim brojem slojeva rotacija i sprezanja.
- Svako kolo uči raspodelu naredne kombinacije.
- Hronološka validacija bira dubinu koja daje najmanji NLL; pri praktično jednakom rezultatu prednost ima jednostavnije kolo.
- Odvojeni test proverava izbor, pa završni trening koristi celu istoriju svakog CSV-a.
- Lokalna simulacija računa verovatnoće bez nasumičnog uzorkovanja.

Svaki dodatni sloj mora da omogući nove zavisnosti i interferenciju, uz proveru da stvarno menja izlaz.
Qiskit model sa izborom dubine kola. Ispitivanje granica složenosti, a izbor prema prediktivnoj validaciji.


Hronološka validacija bira dubinu Qiskit kola prema kvalitetu naučene raspodele. 
Procena memorije i broja operacija pre treninga. 

Sedam registara od po šest kubita, simuliranih zasebno da se ne zauzima memorija za ceo sistem od 42 kubita. 
Njihove amplitude definišu zajedničku raspodelu sedam rastućih brojeva; maksimum će se tražiti iz te raspodele. 
Porediće se dubine od jednog do tri sloja. 
Provere gradijenta i normalizacije ugrađene u kod.
"""
