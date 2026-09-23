import os
import pandas as pd
import numpy as np
import json


class GetTransformData:
    def __init__(self, data_path: str):
        self.matches_path = data_path

    def _time_to_seconds(self, time_str):
        if isinstance(time_str, float):
            return 90 * 60  # 120 minutes = 7200 seconds
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s

    def get_de_data(self, match_id: str) -> pd.DataFrame:
        csv_path = os.path.join(
            self.matches_path, match_id, f"{match_id}_dynamic_events.csv")
        de_match = pd.read_csv(csv_path)
        return de_match

    def get_match_metadata(self, match_id: str) -> pd.DataFrame:
        json_path = os.path.join(
            self.matches_path, match_id, f"{match_id}_match.json"
        )
        with open(json_path, "r") as f:
            data = json.load(f)

        raw_match_df = pd.json_normalize(data, max_level=2)
        raw_match_df["home_team_side"] = raw_match_df["home_team_side"].astype(str)
        players_df = pd.json_normalize(
            raw_match_df.to_dict("records"),
            record_path="players",
            meta=[
                "home_team_score",
                "away_team_score",
                "date_time",
                "home_team_side",
                "home_team.name",
                "home_team.id",
                "away_team.name",
                "away_team.id",
            ],  # data we keep
        )
        # Take only players who played and create their total time
        players_df = players_df[
            ~((players_df.start_time.isna()) & (players_df.end_time.isna()))
        ]
        players_df["total_time"] = (
                players_df["end_time"].apply(self._time_to_seconds()) -
                players_df["start_time"].apply(self._time_to_seconds())
        )

        # Create a flag for GK
        players_df["is_gk"] = players_df["player_role.acronym"] == "GK"

        # Add a flag of the match name
        players_df["match_name"] = (
                players_df["home_team.name"] +
                " vs " +
                players_df["away_team.name"]
        )

        # Add a flag if the given player is home or away
        players_df["home_away_player"] = np.where(
            players_df.team_id == players_df["home_team.id"], "Home", "Away"
        )

        # Create flag from player
        players_df["team_name"] = np.where(
            players_df.team_id == players_df["home_team.id"],
            players_df["home_team.name"],
            players_df["away_team.name"],
        )

        # Figure out sides
        players_df[["home_team_side_1st_half", "home_team_side_2nd_half"]] = (
            players_df["home_team_side"]
            .astype(str)
            .str.strip("[]")
            .str.replace("'", "")
            .str.split(", ", expand=True)
        )
        # Clean up sides
        players_df["direction_player_1st_half"] = np.where(
            players_df.home_away_player == "Home",
            players_df.home_team_side_1st_half,
            players_df.home_team_side_2nd_half,
        )
        players_df["direction_player_2nd_half"] = np.where(
            players_df.home_away_player == "Home",
            players_df.home_team_side_2nd_half,
            players_df.home_team_side_1st_half,
        )

        # Clean up and keep the columns that we want to keep about

        columns_to_keep = [
            "start_time",
            "end_time",
            "match_name",
            "date_time",
            "home_team.name",
            "away_team.name",
            "id",
            "short_name",
            "number",
            "team_id",
            "team_name",
            "player_role.position_group",
            "total_time",
            "player_role.name",
            "player_role.acronym",
            "is_gk",
            "direction_player_1st_half",
            "direction_player_2nd_half",
            "home_away_player"
        ]
        players_df = players_df[columns_to_keep]
        return players_df

    def get_phases_data(self, match_id: str) -> pd.DataFrame:
        csv_path = os.path.join(
            self.matches_path, match_id, f"{match_id}_phases_of_play.csv"
        )
        match_phase = pd.read_csv(csv_path)
        return match_phase

    def get_tracking_data(self, match_id: str) -> pd.DataFrame:
        json_path = os.path.join(
            self.matches_path, match_id, f"{match_id}_tracking_extrapolated.jsonl"
        )
        with open(json_path, "r") as f:
            records = [json.loads(line) for line in f]
        raw_df = pd.json_normalize(records,
                                   "player_data",
                                   ["frame", "timestamp", "period", "possession", "ball_data"])
        raw_df["possession_player_id"] = raw_df["possession"].apply(
            lambda x: x.get("player_id")
        )
        raw_df["possession_group"] = raw_df["possession"].apply(
            lambda x: x.get("group"))

        # Expand the ball_data with json_normalize
        raw_df[["ball_x", "ball_y", "ball_z", "is_detected_ball"]] = pd.json_normalize(
            raw_df.ball_data
        )

        # Expand the image_corners_projection
        raw_df[
            ["x_top_left",
             "y_top_left",
             "x_bottom_left",
             "y_bottom_left",
             "x_bottom_right",
             "y_bottom_right",
             "x_top_right",
             "y_top_right"]
        ] = pd.json_normalize(raw_df.image_corners_projection)

        # Drop the original 'possession' column if you no longer need it
        raw_df = raw_df.drop(columns=["possession", "ball_data"])

        # Add the match_id identifier to your dataframe
        raw_df["match_id"] = match_id
        return raw_df

    def map_tracking_to_phases(
            self,
            track_df: pd.DataFrame,
            phase_df: pd.DataFrame,
            players_df: pd.DataFrame
    ) -> pd.DataFrame:
        # Select top 10 outfield players + 1 goalkeeper per team (starters)
        outfield = players_df[~players_df["is_gk"]].groupby("team_id", group_keys=False).apply(
            lambda x: x.nlargest(10, "total_time")).reset_index(drop=True)
        gk = players_df[players_df["is_gk"]].groupby("team_id", group_keys=False).apply(
            lambda x: x.nlargest(1, "total_time")).reset_index(drop=True)
        selected_players = pd.concat([outfield, gk], ignore_index=True)
        main_df = track_df[track_df["possession_group"].notnull()].copy()
        grouped = main_df.merge(
            selected_players[
                ["team_id", "player_role.name", "id", "first_name", "short_name", "start_time", "end_time", "number",
                 "home_team.name", "away_team.name", "game", "home_away_player", "direction_player_1st_half",
                 "direction_player_2nd_half", "is_gk"]],
            left_on="player_id", right_on="id"
        )
        grouped["direction_player"] = np.where(grouped["period"] == 1, grouped["direction_player_1st_half"],
                                               grouped["direction_player_2nd_half"])
        grouped["x"] = np.where(
            grouped["direction_player"] == "right_to_left",
            -grouped["x"],
            grouped["x"]
        )
        grouped["y"] = np.where(
            grouped["direction_player"] == "right_to_left",
            -grouped["y"],
            grouped["y"]
        )
        grouped["team"] = np.where(
            grouped["home_away_player"] == "Home",
            grouped["home_team.name"],
            grouped["away_team.name"])

        interval_index = pd.IntervalIndex.from_arrays(phase_df["frame_start"], phase_df["frame_end"], closed="left")
        matched_idx = interval_index.get_indexer(grouped["frame"])

        valid_mask = matched_idx != -1
        tracking_valid = grouped[valid_mask].copy().reset_index(drop=True)
        matched_phases = phase_df.iloc[matched_idx[valid_mask]].reset_index(drop=True)

        combined = tracking_valid.join(matched_phases[["team_in_possession_id", "team_in_possession_phase_type",
                                                       "team_out_of_possession_phase_type"]])
        combined["phase"] = np.where(
            combined["team_id"] == combined["team_in_possession_id"],
            combined["team_in_possession_phase_type"],
            combined["team_out_of_possession_phase_type"]
        )
        return combined

    def merge_tracking_metadata(self, track_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
        temp_df = track_df.merge(players_df, left_on="player_id", right_on="id")
        # 2. Assign direction based on the current period
        # Uses the period as an index to pick between 1st and 2nd half columns
        temp_df['current_direction'] = np.where(
            temp_df['period'] == 1,
            temp_df['direction_player_1st_half'],
            temp_df['direction_player_2nd_half']
        )

        # 3. Determine possession status
        # We simplify the boolean check by matching the team labels
        is_home_poss = (temp_df['possession_group'] == 'home team') & (temp_df['home_away_player'] == 'Home')
        is_away_poss = (temp_df['possession_group'] == 'away team') & (temp_df['home_away_player'] == 'Away')
        temp_df['in_possession'] = is_home_poss | is_away_poss

        # 4. Create the Flip Mask
        # We flip if:
        # (Attacking right-to-left AND in possession) OR (Defending left-to-right AND out of possession)
        flip_mask = (
                ((temp_df['current_direction'] == 'right_to_left') & temp_df['in_possession']) |
                ((temp_df['current_direction'] == 'left_to_right') & ~temp_df['in_possession'])
        )

        # 5. Apply coordinate inversion
        # Iterating over existing columns avoids repetitive 'np.where' blocks
        coords = [c for c in ['x', 'y', 'ball_x', 'ball_y'] if c in temp_df.columns]
        for col in coords:
            temp_df[col] = np.where(flip_mask, -temp_df[col], temp_df[col])

        enriched_tracking_data = temp_df
        return enriched_tracking_data
