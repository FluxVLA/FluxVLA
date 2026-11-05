# Copyright 2026 Limx Dynamics
"""Explicit, finite VLA capabilities; never inferred from weights."""

import copy
import itertools
import re
import string
from dataclasses import dataclass
from typing import Dict, Mapping, Optional


def validate_conditions(conditions, name, allow_empty=False):
    if not isinstance(conditions, Mapping) or (not conditions
                                               and not allow_empty):
        raise ValueError(f'{name} must be a nonempty predicate mapping')
    if any(not isinstance(k, str) or not k or type(v) is not bool
           for k, v in conditions.items()):
        raise ValueError(
            f'{name} requires nonempty string keys and bool values')
    return dict(conditions)


def matches(conditions, facts):
    """Unknown/missing is NOT false. In particular, it cannot prove closure."""
    return all(
        facts.get(key) is expected for key, expected in conditions.items())


def has_unknown(conditions, facts):
    return any(type(facts.get(key)) is not bool for key in conditions)


def _template(text, fields):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Templates must be nonempty strings')
    for _, field, spec, conversion in string.Formatter().parse(text):
        if field is not None and (field not in fields or spec or conversion):
            raise ValueError(f'Unsupported template field: {field!r}')
    return text


def _bind_conditions(conditions, arguments):
    result = {}
    for template, expected in conditions.items():
        key = template.format_map(arguments)
        if key in result:
            raise ValueError(
                f'Predicate templates collide after binding: {key}')
        result[key] = expected
    return result


@dataclass(frozen=True)
class Candidate:
    choice: str
    skill: str
    arguments: Dict[str, str]
    instruction: str
    preconditions: Dict[str, bool]
    completion: Dict[str, bool]
    max_chunks: int

    def decision(self, kind='execute'):
        return dict(
            decision=kind,
            skill=self.skill,
            arguments=copy.deepcopy(self.arguments))

    def describe(self):
        return dict(
            choice=self.choice,
            **self.decision(),
            instruction=self.instruction)


class SkillCatalog:
    """Expand parameter enums into a bounded set of grounded candidates.

    ``entities`` maps stable ids to VLA-readable descriptions. Each skill
    declares ``name``, ``parameters`` (id enums), ``instruction``, and boolean
    ``completion`` predicates. Predicate key templates use ids; instruction
    templates use descriptions. This keeps the model out of string rendering.
    """

    def __init__(self, entities, skills, max_candidates=128):
        if not isinstance(entities, Mapping) or any(
                not isinstance(k, str) or not k or not isinstance(v, str)
                or not v.strip() for k, v in entities.items()):
            raise ValueError('entities must map ids to nonempty descriptions')
        if type(max_candidates) is not int or max_candidates < 1:
            raise ValueError('max_candidates must be a positive integer')
        if not isinstance(skills, (list, tuple)) or not skills:
            raise ValueError('skills must be a nonempty list')
        self.entities = dict(entities)
        self.candidates = {}
        names = set()
        for raw in skills:
            spec = copy.deepcopy(dict(raw))
            name = spec.pop('name')
            if (not isinstance(name, str)
                    or not re.fullmatch(r'[a-z][a-z0-9_]*', name)
                    or name in names):
                raise ValueError(f'Invalid or duplicate skill name: {name!r}')
            names.add(name)
            parameters = spec.pop('parameters', {})
            if not isinstance(parameters, Mapping):
                raise ValueError('parameters must be a mapping of id enums')
            combinations = 1
            for param, values in parameters.items():
                if not isinstance(param, str) or not param.isidentifier():
                    raise ValueError(f'Invalid parameter: {param!r}')
                if (not isinstance(values, (list, tuple)) or not values
                        or any(not isinstance(v, str) or v not in entities
                               for v in values)
                        or len(set(values)) != len(values)):
                    raise ValueError(
                        f'{param} must enumerate unique entity ids')
                combinations *= len(values)
            if len(self.candidates) + combinations > max_candidates:
                raise ValueError('Skill expansion exceeds max_candidates')
            instruction = _template(spec.pop('instruction'), parameters)
            preconditions = validate_conditions(
                spec.pop('preconditions', {}), 'preconditions', True)
            completion = validate_conditions(
                spec.pop('completion'), 'completion')
            for key in itertools.chain(preconditions, completion):
                _template(key, parameters)
            max_chunks = spec.pop('max_chunks', 50)
            if type(max_chunks) is not int or max_chunks < 1:
                raise ValueError('max_chunks must be a positive integer')
            if spec:
                raise ValueError(f'Unknown skill fields: {sorted(spec)}')
            for values in itertools.product(*parameters.values()):
                arguments = dict(zip(parameters, values))
                labels = {k: entities[v] for k, v in arguments.items()}
                choice = f'skill_{len(self.candidates):04d}'
                self.candidates[choice] = Candidate(
                    choice=choice,
                    skill=name,
                    arguments=arguments,
                    instruction=instruction.format_map(labels),
                    preconditions=_bind_conditions(preconditions, arguments),
                    completion=_bind_conditions(completion, arguments),
                    max_chunks=max_chunks)

    @property
    def predicate_keys(self):
        return set().union(*(set(c.preconditions) | set(c.completion)
                             for c in self.candidates.values()))

    def available(self, facts):
        return [
            c for c in self.candidates.values()
            if matches(c.preconditions, facts) and not has_unknown(
                c.completion, facts) and not matches(c.completion, facts)
        ]

    def resolve(self, choice) -> Optional[Candidate]:
        return self.candidates.get(choice)


def choice_schema(choices):
    """The model only selects an id; the catalog constructs the full JSON."""
    return dict(
        type='object',
        properties={'choice': dict(type='string', enum=list(choices))},
        required=['choice'],
        additionalProperties=False)


def validate_choice(value, choices):
    if (type(value) is not dict or set(value) != {'choice'}
            or type(value['choice']) is not str
            or value['choice'] not in choices):
        raise ValueError('Planner must return exactly one allowed choice')
    return value['choice']
