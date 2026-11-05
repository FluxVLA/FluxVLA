# Copyright 2026 Limx Dynamics
"""Offline replay: no model, network, ROS or robot commands.

Skill names and predicates are illustrative, not capabilities certified for
any checkpoint. Replayed scene changes are fixtures, not executed actions.
"""

inference = dict(
    type='HierarchicalInferenceRunner',
    # Optional paired defaults for this fixed replay, not a runner requirement.
    # Real tasks pass task_description and goal_conditions to run() / the CLI.
    task_description=('Put the red cup in the top drawer, '
                      'then close the drawer.'),
    dry_run=True,
    stable_observations=2,
    max_steps=20,
    skill_catalog=dict(
        entities=dict(cup='red cup', drawer='top drawer'),
        skills=[
            dict(
                name='open_drawer',
                parameters=dict(drawer=['drawer']),
                instruction='Open the {drawer}.',
                preconditions={'open:{drawer}': False},
                completion={'open:{drawer}': True},
                max_chunks=10),
            dict(
                name='put_into',
                parameters=dict(object=['cup'], target=['drawer']),
                instruction='Put the {object} into the {target}.',
                preconditions={'open:{target}': True},
                completion={'inside:{object}:{target}': True},
                max_chunks=20),
            dict(
                name='close_drawer',
                parameters=dict(drawer=['drawer']),
                instruction='Close the {drawer}.',
                preconditions={
                    'open:{drawer}': True,
                    'inside:cup:{drawer}': True
                },
                completion={'open:{drawer}': False},
                max_chunks=10),
        ]),
    planner=dict(
        type='ScriptedSubtaskPlanner',
        choices=['skill_0000', 'skill_0001', 'skill_0002']),
    monitor=dict(type='ObservationPredicateMonitor'),
    goal_conditions={
        'inside:cup:drawer': True,
        'open:drawer': False
    },
    observations=[
        dict(predicates={
            'open:drawer': False,
            'inside:cup:drawer': False
        }),
        dict(predicates={
            'open:drawer': True,
            'inside:cup:drawer': False
        }),
        dict(predicates={
            'open:drawer': True,
            'inside:cup:drawer': False
        }),
        dict(predicates={
            'open:drawer': True,
            'inside:cup:drawer': True
        }),
        dict(predicates={
            'open:drawer': True,
            'inside:cup:drawer': True
        }),
        dict(predicates={
            'open:drawer': False,
            'inside:cup:drawer': True
        }),
        dict(predicates={
            'open:drawer': False,
            'inside:cup:drawer': True
        }),
    ],
)
