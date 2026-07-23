import unittest

from vertical_focus.config import RuntimeParams, load_configuration
from vertical_focus.policy import FocusPolicy, MovieFocusPolicy, SportsFocusPolicy
from vertical_focus.types import Detection, TrackView


SPORTS_CONFIG, _, _ = load_configuration(RuntimeParams(mode="sports"))
MOVIE_CONFIG, _, _ = load_configuration(RuntimeParams(mode="movie"))


def track(
    track_id: str,
    label: str,
    score: float,
    box,
    *,
    hits: int = 5,
    misses: int = 0,
    observed: bool = True,
) -> TrackView:
    return TrackView(
        track_id,
        label,
        score,
        box,
        hits=hits,
        misses=misses,
        observed_this_frame=observed,
    )


def motion(center_x: float = 0.50, score: float = 0.30) -> Detection:
    return Detection(
        "motion",
        score,
        (max(0.0, center_x - 0.12), 0.25, min(1.0, center_x + 0.12), 0.85),
    )


def gameplay_people(offset: float = 0.0):
    return [
        track(
            f"person:{index}",
            "person",
            0.82,
            (0.08 + index * 0.14 + offset, 0.28, 0.18 + index * 0.14 + offset, 0.92),
        )
        for index in range(5)
    ]


def character(person_id: int, center_x: float, *, face: bool = True, face_score: float = 0.90):
    person = track(
        f"person:{person_id}",
        "person",
        0.90,
        (center_x - 0.10, 0.16, center_x + 0.10, 0.92),
    )
    rows = [person]
    if face:
        rows.append(
            track(
                f"face:{person_id}",
                "face",
                face_score,
                (center_x - 0.045, 0.18, center_x + 0.045, 0.38),
            )
        )
    return rows


class PolicySeparationTests(unittest.TestCase):
    def test_modes_use_distinct_policy_implementations(self):
        sports = FocusPolicy("sports", SPORTS_CONFIG)
        movie = FocusPolicy("movie", MOVIE_CONFIG)
        self.assertIsInstance(sports.implementation, SportsFocusPolicy)
        self.assertIsInstance(movie.implementation, MovieFocusPolicy)
        self.assertIsNot(type(sports.implementation), type(movie.implementation))


class SportsPolicyTests(unittest.TestCase):
    def test_ball_is_dominant_and_player_is_context_only(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        player = track("person:1", "person", 0.92, (0.40, 0.22, 0.56, 0.94))
        ball = track("ball:1", "ball", 0.78, (0.62, 0.48, 0.64, 0.51), hits=4)
        selected = policy.select([player, ball], None, crop_width=0.316, time_s=0.0)
        ball_center = 0.63
        player_center = 0.48
        self.assertEqual("ball_focus", selected.label)
        self.assertEqual("observed_ball", selected.evidence_kind)
        self.assertLess(abs(selected.center_x - ball_center), abs(selected.center_x - player_center))
        self.assertLessEqual(abs(selected.center_x - ball_center), 0.04)
        self.assertEqual(1, len(selected.boxes))
        self.assertEqual("ball", selected.boxes[0].label)

    def test_far_low_confidence_ball_does_not_replace_fresh_path(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        people = gameplay_people()
        real = track("ball:1", "ball", 0.80, (0.23, 0.48, 0.25, 0.51), hits=4)
        first = policy.select(people + [real], motion(0.25), crop_width=0.316, time_s=0.0)
        self.assertEqual("sports_ball", first.key)

        false_ball = track("ball:9", "ball", 0.27, (0.61, 0.46, 0.63, 0.49), hits=3)
        second = policy.select(people + [false_ball], motion(0.62), crop_width=0.316, time_s=0.10)
        self.assertEqual("sports_ball", second.key)
        self.assertEqual("predicted_ball", second.evidence_kind)
        self.assertLess(second.center_x, 0.35)


    def test_low_confidence_static_ball_without_close_context_is_not_trusted(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        static_ball = TrackView(
            "ball:1", "ball", 0.27, (0.58, 0.46, 0.60, 0.49), hits=6,
            vx_per_frame=0.0, vy_per_frame=0.0, observed_this_frame=True
        )
        selected = policy.select([static_ball], None, crop_width=0.316, time_s=0.0)
        self.assertNotEqual("ball_focus", selected.label)
        self.assertEqual("locked", selected.smoothing_regime)

    def test_low_confidence_static_ball_near_player_is_not_trusted(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        player = track("person:1", "person", 0.94, (0.48, 0.22, 0.66, 0.94))
        static_ball = TrackView(
            "ball:1",
            "ball",
            0.27,
            (0.56, 0.46, 0.58, 0.49),
            hits=8,
            vx_per_frame=0.0,
            vy_per_frame=0.0,
            observed_this_frame=True,
        )
        selected = policy.select(
            [player, static_ball], None, crop_width=0.316, time_s=0.0
        )
        self.assertNotEqual("ball_focus", selected.label)
        self.assertEqual("locked", selected.smoothing_regime)

    def test_low_confidence_consistently_moving_ball_can_be_trusted(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        moving_ball = TrackView(
            "ball:1", "ball", 0.27, (0.58, 0.46, 0.60, 0.49), hits=4,
            vx_per_frame=0.003, vy_per_frame=-0.001, observed_this_frame=True
        )
        selected = policy.select([moving_ball], None, crop_width=0.316, time_s=0.0)
        self.assertEqual("ball_focus", selected.label)

    def test_confirmed_near_path_reacquisition_enables_fast_regime(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        people = gameplay_people()
        first_ball = track("ball:1", "ball", 0.82, (0.23, 0.48, 0.25, 0.51), hits=4)
        first = policy.select(people + [first_ball], motion(0.25), crop_width=0.316, time_s=0.0)
        self.assertEqual("ball_follow", first.smoothing_regime)

        reacquired = track("ball:2", "ball", 0.83, (0.38, 0.47, 0.40, 0.50), hits=2)
        second = policy.select(people + [reacquired], motion(0.39), crop_width=0.316, time_s=0.25)
        self.assertEqual("reacquired_ball", second.evidence_kind)
        self.assertEqual("fast_reacquire", second.smoothing_regime)

    def test_gameplay_state_requires_persistent_people_and_motion(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        people = gameplay_people()
        first = policy.select(people, motion(0.48), crop_width=0.316, time_s=0.0)
        self.assertEqual("safe_center", first.label)
        self.assertEqual("weak_evidence", first.scene_state)
        second = policy.select(people, motion(0.50), crop_width=0.316, time_s=0.30)
        self.assertEqual("gameplay_cluster", second.label)
        self.assertEqual("live_gameplay", second.scene_state)

    def test_gameplay_cluster_key_is_stable_across_membership_change(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        people_a = gameplay_people()
        policy.select(people_a, motion(0.48), crop_width=0.316, time_s=0.0)
        first = policy.select(people_a, motion(0.48), crop_width=0.316, time_s=0.30)
        people_b = people_a[1:] + [
            track("person:9", "person", 0.83, (0.70, 0.28, 0.80, 0.92))
        ]
        second = policy.select(people_b, motion(0.54), crop_width=0.316, time_s=0.50)
        self.assertEqual("gameplay_cluster", first.key)
        self.assertEqual(first.key, second.key)

    def test_live_gameplay_does_not_follow_announcer_face(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        people = gameplay_people()
        large_face = track("face:99", "face", 0.99, (0.72, 0.08, 0.94, 0.48))
        policy.select(people + [large_face], motion(0.48), crop_width=0.316, time_s=0.0)
        selected = policy.select(
            people + [large_face], motion(0.50), crop_width=0.316, time_s=0.30
        )
        self.assertEqual("gameplay_cluster", selected.label)


    def test_many_static_people_become_timeout_not_gameplay(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        people = gameplay_people()
        first = policy.select(people, None, crop_width=0.316, time_s=0.0)
        second = policy.select(people, None, crop_width=0.316, time_s=0.40)
        self.assertNotEqual("gameplay_cluster", first.label)
        self.assertEqual("timeout_or_bench", second.scene_state)
        self.assertEqual("locked", second.smoothing_regime)

    def test_small_single_person_is_not_chased(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        small = track("person:1", "person", 0.99, (0.72, 0.36, 0.77, 0.64), hits=8)
        first = policy.select([small], None, crop_width=0.316, time_s=0.0)
        second = policy.select([small], None, crop_width=0.316, time_s=0.40)
        self.assertEqual("safe_center", first.label)
        self.assertEqual("safe_center", second.label)
        self.assertEqual("locked", second.smoothing_regime)

    def test_no_evidence_holds_exact_last_composition(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        ball = track("ball:1", "ball", 0.85, (0.70, 0.48, 0.72, 0.51), hits=4)
        first = policy.select(gameplay_people() + [ball], motion(0.71), crop_width=0.316, time_s=0.0)
        held = policy.select([], None, crop_width=0.316, time_s=1.20)
        self.assertEqual(first.center_x, held.center_x)
        self.assertEqual("locked", held.smoothing_regime)
        self.assertEqual((), held.boxes)
        self.assertNotEqual("ball_focus", held.label)

    def test_large_low_confidence_moving_ball_inside_player_beats_small_artifact(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        player = track("person:1", "person", 0.90, (0.08, 0.20, 0.22, 0.94))
        real_ball = TrackView(
            "ball:2",
            "ball",
            0.10,
            (0.12, 0.60, 0.14, 0.63),
            hits=3,
            vx_per_frame=-0.0016,
            vy_per_frame=-0.0038,
            observed_this_frame=True,
        )
        artifact = TrackView(
            "ball:1",
            "ball",
            0.27,
            (0.66, 0.49, 0.67, 0.505),
            hits=5,
            vx_per_frame=-0.001,
            vy_per_frame=0.0,
            observed_this_frame=True,
        )
        selected = policy.select(
            gameplay_people() + [player, real_ball, artifact],
            motion(0.50),
            crop_width=0.316,
            time_s=0.0,
        )
        self.assertEqual("ball_focus", selected.label)
        self.assertEqual("ball:2", selected.boxes[0].focus_id)
        self.assertLess(selected.center_x, 0.25)

    def test_high_confidence_tight_context_ball_replaces_stale_path_on_first_hit(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        stale_ball = track(
            "ball:1", "ball", 0.82, (0.20, 0.46, 0.22, 0.49), hits=5
        )
        first = policy.select(
            gameplay_people() + [stale_ball],
            motion(0.21),
            crop_width=0.316,
            time_s=0.0,
        )
        self.assertEqual("ball:1", first.boxes[0].focus_id)

        interacting_player = track(
            "person:20", "person", 0.92, (0.10, 0.18, 0.28, 0.94)
        )
        observed_ball = TrackView(
            "ball:2",
            "ball",
            0.32,
            (0.17, 0.34, 0.19, 0.37),
            hits=1,
            observed_this_frame=True,
        )
        selected = policy.select(
            gameplay_people() + [interacting_player, observed_ball],
            motion(0.18),
            crop_width=0.316,
            time_s=0.80,
        )
        self.assertEqual("ball:2", selected.boxes[0].focus_id)
        self.assertEqual("observed_ball", selected.evidence_kind)
        self.assertLess(selected.center_x, 0.25)

    def test_broad_static_player_proximity_does_not_create_new_ball_lock(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        initial = track(
            "ball:1", "ball", 0.82, (0.20, 0.46, 0.22, 0.49), hits=5
        )
        policy.select(
            gameplay_people() + [initial],
            motion(0.21),
            crop_width=0.316,
            time_s=0.0,
        )
        nearby_player = track(
            "person:20", "person", 0.90, (0.62, 0.18, 0.76, 0.94)
        )
        static_artifact = TrackView(
            "ball:9",
            "ball",
            0.35,
            (0.54, 0.52, 0.555, 0.541),
            hits=9,
            vx_per_frame=0.0002,
            observed_this_frame=True,
        )
        selected = policy.select(
            [nearby_player, static_artifact],
            motion(0.55),
            crop_width=0.316,
            time_s=0.80,
        )
        self.assertNotEqual("ball:9", selected.boxes[0].focus_id)
        self.assertNotEqual("ball_focus", selected.label)

    def test_contextual_ball_reacquires_over_fresh_false_path_after_persistence(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        people = gameplay_people()
        false_ball = TrackView(
            "ball:1",
            "ball",
            0.31,
            (0.58, 0.48, 0.595, 0.495),
            hits=5,
            vx_per_frame=0.002,
            observed_this_frame=True,
        )
        first = policy.select(
            people + [false_ball], motion(0.59), crop_width=0.316, time_s=0.0
        )
        self.assertEqual("ball:1", first.boxes[0].focus_id)

        context_player = track(
            "person:20", "person", 0.90, (0.10, 0.18, 0.28, 0.94)
        )
        true_ball = TrackView(
            "ball:2",
            "ball",
            0.28,
            (0.17, 0.34, 0.19, 0.37),
            hits=3,
            vx_per_frame=-0.001,
            vy_per_frame=0.001,
            observed_this_frame=True,
        )
        second = policy.select(
            people + [context_player, true_ball],
            motion(0.18),
            crop_width=0.316,
            time_s=0.25,
        )
        self.assertEqual("ball:2", second.boxes[0].focus_id)
        self.assertEqual("reacquired_ball", second.evidence_kind)

    def test_predicted_ball_is_not_mislabeled_as_reacquired(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        ball = track(
            "ball:1", "ball", 0.82, (0.30, 0.45, 0.32, 0.48), hits=4
        )
        first = policy.select(
            gameplay_people() + [ball],
            motion(0.31),
            crop_width=0.316,
            time_s=0.0,
        )
        self.assertEqual("observed_ball", first.evidence_kind)

        predicted = TrackView(
            "ball:1",
            "ball",
            0.78,
            (0.31, 0.45, 0.33, 0.48),
            hits=4,
            misses=1,
            vx_per_frame=0.002,
            observed_this_frame=False,
        )
        second = policy.select(
            gameplay_people() + [predicted],
            motion(0.32),
            crop_width=0.316,
            time_s=0.20,
        )
        self.assertEqual("predicted_ball", second.evidence_kind)
        self.assertNotEqual("fast_reacquire", second.smoothing_regime)

    def test_gameplay_region_ignores_one_frame_target_jump(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        people = gameplay_people()
        policy.select(people, motion(0.30), crop_width=0.316, time_s=0.0)
        first = policy.select(
            people, motion(0.30), crop_width=0.316, time_s=0.30
        )
        jumped = policy.select(
            people, motion(0.80), crop_width=0.316, time_s=0.40
        )
        self.assertEqual("gameplay_cluster", first.label)
        self.assertAlmostEqual(first.center_x, jumped.center_x, places=6)

    def test_bottom_border_false_ball_cannot_replace_plausible_path_ball(self):
        policy = FocusPolicy("sports", SPORTS_CONFIG)
        initial_ball = TrackView(
            "ball:1",
            "ball",
            0.52,
            (0.294, 0.287, 0.316, 0.322),
            hits=5,
            vx_per_frame=0.0015,
            vy_per_frame=-0.0010,
            observed_this_frame=True,
        )
        policy.select(
            gameplay_people() + [initial_ball],
            motion(0.31),
            crop_width=0.316,
            time_s=0.0,
        )

        active_player = track(
            "person:20", "person", 0.88, (0.34, 0.18, 0.48, 0.92), hits=12
        )
        clipped_bottom_person = track(
            "person:21", "person", 0.50, (0.779, 0.941, 0.824, 0.996), hits=6
        )
        plausible_ball = TrackView(
            "ball:2",
            "ball",
            0.134,
            (0.3975, 0.2332, 0.4113, 0.2571),
            hits=5,
            vx_per_frame=0.0020,
            vy_per_frame=-0.0017,
            observed_this_frame=True,
        )
        border_false_ball = TrackView(
            "ball:3",
            "ball",
            0.64,
            (0.7700, 0.9379, 0.7950, 0.9913),
            hits=2,
            vx_per_frame=-0.0002,
            vy_per_frame=-0.0016,
            observed_this_frame=True,
        )
        selected = policy.select(
            gameplay_people()
            + [active_player, clipped_bottom_person, plausible_ball, border_false_ball],
            motion(0.40),
            crop_width=0.316,
            time_s=0.20,
        )
        self.assertEqual("ball_focus", selected.label)
        self.assertEqual("ball:2", selected.boxes[0].focus_id)
        self.assertLess(selected.center_x, 0.50)


class MoviePolicyTests(unittest.TestCase):
    def test_face_is_anchored_to_enclosing_person_identity(self):
        policy = FocusPolicy("movie", MOVIE_CONFIG)
        selected = policy.select(character(1, 0.40), None, crop_width=0.316, time_s=0.0)
        self.assertEqual("character:person:1", selected.key)
        self.assertEqual("primary_face", selected.label)
        self.assertEqual("frontal_face", selected.evidence_kind)
        self.assertEqual(1, len(selected.boxes))
        self.assertEqual("person", selected.boxes[0].label)

    def test_face_loss_falls_back_to_same_person(self):
        policy = FocusPolicy("movie", MOVIE_CONFIG)
        first = policy.select(character(1, 0.40), None, crop_width=0.316, time_s=0.0)
        second = policy.select(character(1, 0.40, face=False), None, crop_width=0.316, time_s=0.20)
        self.assertEqual(first.key, second.key)
        self.assertEqual("primary_person", second.label)

    def test_tiny_background_face_does_not_preempt_current_character(self):
        policy = FocusPolicy("movie", MOVIE_CONFIG)
        primary = character(1, 0.38)
        first = policy.select(primary, None, crop_width=0.316, time_s=0.0)
        poster_person = track("person:9", "person", 0.99, (0.76, 0.28, 0.86, 0.48))
        poster_face = track("face:9", "face", 0.99, (0.80, 0.30, 0.84, 0.34), hits=8)
        selected = first
        for timestamp in (1.0, 1.3, 1.6, 2.0):
            selected = policy.select(
                primary + [poster_person, poster_face],
                None,
                crop_width=0.316,
                time_s=timestamp,
            )
        self.assertEqual(first.key, selected.key)

    def test_nonfrontal_head_does_not_preempt_frontal_character(self):
        policy = FocusPolicy("movie", MOVIE_CONFIG)
        primary = character(1, 0.38)
        first = policy.select(primary, None, crop_width=0.316, time_s=0.0)
        second_person = track("person:2", "person", 0.98, (0.64, 0.12, 0.86, 0.94))
        second_head = track("head:2", "head", 0.99, (0.69, 0.16, 0.81, 0.40), hits=8)
        selected = policy.select(
            primary + [second_person, second_head],
            None,
            crop_width=0.316,
            time_s=1.2,
        )
        self.assertEqual(first.key, selected.key)

    def test_motion_cannot_compete_with_valid_character(self):
        policy = FocusPolicy("movie", MOVIE_CONFIG)
        selected = policy.select(
            character(1, 0.38), motion(0.85, score=0.99), crop_width=0.316, time_s=0.0
        )
        self.assertTrue(selected.key.startswith("character:"))
        self.assertNotEqual("action_region", selected.label)

    def test_no_evidence_holds_exact_position_and_zero_motion_regime(self):
        policy = FocusPolicy("movie", MOVIE_CONFIG)
        first = policy.select(character(1, 0.68), None, crop_width=0.316, time_s=0.0)
        second = policy.select([], None, crop_width=0.316, time_s=2.0)
        self.assertEqual(first.key, second.key)
        self.assertEqual(first.center_x, second.center_x)
        self.assertEqual("locked", second.smoothing_regime)
        self.assertEqual((), second.boxes)

    def test_single_frame_subject_gap_holds_person_box_without_long_stale_box(self):
        policy = FocusPolicy("movie", MOVIE_CONFIG)
        first = policy.select(
            character(1, 0.52), None, crop_width=0.316, time_s=0.0
        )
        short_gap = policy.select(
            [], None, crop_width=0.316, time_s=0.10
        )
        self.assertEqual("short_focus_box_hold", short_gap.evidence_kind)
        self.assertEqual(first.boxes, short_gap.boxes)

        long_gap = policy.select(
            [], None, crop_width=0.316, time_s=0.30
        )
        self.assertEqual("no_evidence_hold", long_gap.evidence_kind)
        self.assertEqual((), long_gap.boxes)

    def test_missing_cut_can_replace_lost_character_with_strong_frontal_face(self):
        policy = FocusPolicy("movie", MOVIE_CONFIG)
        first = policy.select(
            character(1, 0.28), None, crop_width=0.316, time_s=0.0
        )
        replacement = policy.select(
            character(2, 0.72), None, crop_width=0.316, time_s=0.20
        )
        self.assertNotEqual(first.key, replacement.key)
        self.assertEqual("character:person:2", replacement.key)
        self.assertEqual("frontal_face", replacement.evidence_kind)
        self.assertEqual("confirmed_reframe", replacement.smoothing_regime)

    def test_oversized_person_only_candidate_does_not_beat_frontal_face(self):
        policy = FocusPolicy("movie", MOVIE_CONFIG)
        oversized = track(
            "person:9", "person", 0.99, (0.03, 0.05, 0.97, 0.97), hits=8
        )
        foreground = character(2, 0.70)
        selected = policy.select(
            [oversized] + foreground, None, crop_width=0.316, time_s=0.0
        )
        self.assertEqual("character:person:2", selected.key)
        self.assertEqual("primary_face", selected.label)

    def test_overlapping_characters_receive_one_face_each(self):
        implementation = MovieFocusPolicy(MOVIE_CONFIG)
        persons = [character(1, 0.43)[0], character(2, 0.56)[0]]
        faces = [character(1, 0.43)[1], character(2, 0.56)[1]]
        paired = implementation._pair_regions(faces, persons)
        self.assertEqual({"person:1", "person:2"}, set(paired))
        self.assertEqual("face:1", paired["person:1"].track_id)
        self.assertEqual("face:2", paired["person:2"].track_id)

    def test_stable_character_locks_then_real_movement_unlocks(self):
        policy = FocusPolicy("movie", MOVIE_CONFIG)
        policy.select(character(1, 0.40), None, crop_width=0.316, time_s=0.0)
        stable = policy.select(character(1, 0.40), None, crop_width=0.316, time_s=0.35)
        self.assertEqual("settle_then_lock", stable.smoothing_regime)

        pending = policy.select(character(1, 0.49), None, crop_width=0.316, time_s=0.45)
        self.assertEqual("settle_then_lock", pending.smoothing_regime)
        still_pending = policy.select(
            character(1, 0.50), None, crop_width=0.316, time_s=0.65
        )
        self.assertEqual("settle_then_lock", still_pending.smoothing_regime)
        moved = policy.select(
            character(1, 0.50), None, crop_width=0.316, time_s=0.75
        )
        self.assertEqual("confirmed_reframe", moved.smoothing_regime)


    def test_cut_initializes_from_first_observation_instead_of_safe_center(self):
        policy = FocusPolicy("movie", MOVIE_CONFIG)
        person = track(
            "person:1",
            "person",
            0.90,
            (0.18, 0.16, 0.38, 0.92),
            hits=1,
        )
        face = track(
            "face:1",
            "face",
            0.92,
            (0.22, 0.18, 0.31, 0.38),
            hits=1,
        )
        selected = policy.select(
            [person, face], None, crop_width=0.316, time_s=0.0
        )
        self.assertEqual("primary_face", selected.label)
        self.assertEqual("cut_reset", selected.smoothing_regime)
        self.assertLess(selected.center_x, 0.40)

    def test_safe_center_does_not_delay_first_credible_cut_subject(self):
        policy = FocusPolicy("movie", MOVIE_CONFIG)
        first = policy.select([], None, crop_width=0.316, time_s=0.0)
        self.assertEqual("safe_center", first.label)

        second = policy.select(
            character(1, 0.28), None, crop_width=0.316, time_s=0.20
        )
        self.assertEqual("primary_face", second.label)
        self.assertEqual("cut_reset", second.smoothing_regime)
        self.assertLess(second.center_x, 0.40)

    def test_face_center_is_blended_with_person_anchor(self):
        policy = FocusPolicy("movie", MOVIE_CONFIG)
        person = track(
            "person:1", "person", 0.90, (0.20, 0.16, 0.50, 0.92), hits=5
        )
        face = track(
            "face:1", "face", 0.92, (0.20, 0.18, 0.30, 0.38), hits=5
        )
        selected = policy.select(
            [person, face], None, crop_width=0.316, time_s=0.0
        )
        face_x = 0.25
        person_x = 0.35
        self.assertGreater(selected.center_x, face_x)
        self.assertLess(selected.center_x, person_x)

    def test_interaction_group_key_is_anchored_not_full_membership(self):
        policy = FocusPolicy("movie", MOVIE_CONFIG)
        first_tracks = character(1, 0.43) + character(2, 0.56)
        first = policy.select(first_tracks, None, crop_width=0.40, time_s=0.0)
        self.assertEqual("interaction_group", first.label)
        self.assertTrue(first.key.startswith("interaction_group:person:"))

        second_tracks = character(1, 0.43) + character(3, 0.55)
        second = policy.select(second_tracks, None, crop_width=0.40, time_s=0.3)
        self.assertEqual(first.key, second.key)

    def test_fragmented_person_track_relinks_to_current_character(self):
        policy = FocusPolicy("movie", MOVIE_CONFIG)
        first = policy.select(
            character(1, 0.40), None, crop_width=0.316, time_s=0.0
        )
        replacement_person = track(
            "person:99", "person", 0.91, (0.302, 0.16, 0.502, 0.92), hits=5
        )
        replacement_face = track(
            "face:99", "face", 0.91, (0.356, 0.18, 0.446, 0.38), hits=5
        )
        second = policy.select(
            [replacement_person, replacement_face],
            None,
            crop_width=0.316,
            time_s=0.20,
        )
        self.assertEqual(first.key, second.key)
        self.assertEqual("primary_face", second.label)
        self.assertEqual("person:99", second.boxes[0].focus_id)


if __name__ == "__main__":
    unittest.main()
