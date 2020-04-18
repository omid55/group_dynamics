# Omid55
# Test module for group dynamics logs library.


from __future__ import division, print_function, absolute_import, unicode_literals

import os
import unittest
import numpy as np
import pandas as pd
from pandas import testing as pd_testing
from numpy import testing as np_testing
from parameterized import parameterized
import groupdynamics_logs_lib as gll
import utils


class TestTeamLogsLoaderLoad(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.loader = gll.TeamLogsLoader(
            directory=os.getcwd() + '/src/testing_log/with_confidence')
        cls.loader_no_confidence = gll.TeamLogsLoader(
            directory=os.getcwd() + '/src/testing_log/without_confidence')

    @classmethod
    def tearDown(cls):
        del cls.loader
        del cls.loader_no_confidence

    # =========================================================================
    # ================================ _load ==================================
    # =========================================================================
    def test_load_answers_are_correct(self):
        expected_answers = pd.DataFrame({
            "sender":{0:"pogs10.1",1:"pogs10.1",2:"pogs10.2",3:"pogs10.2",4:"pogs10.1",5:"pogs10.1",6:"pogs10.2",7:"pogs10.2",8:"pogs10.2",9:"pogs10.1",10:"pogs10.1",11:"pogs10.2",12:"pogs10.2",13:"pogs10.2",14:"pogs10.1",15:"pogs10.2",16:"pogs10.2"},
            "question":{0:"GD_solo_surgery0",1:"GD_solo_surgery0",2:"GD_solo_surgery0",3:"GD_solo_surgery0",4:"GD_solo_surgery1",5:"GD_solo_surgery1",6:"GD_solo_surgery1",7:"GD_solo_surgery2",8:"GD_solo_surgery2",9:"GD_solo_surgery3",10:"GD_solo_surgery3",11:"GD_solo_surgery3",12:"GD_solo_surgery3",13:"GD_solo_surgery3",14:"GD_solo_sports0",15:"GD_solo_sports0",16:"GD_solo_sports0"},
            "input":{0:"answer",1:"confidence",2:"answer",3:"confidence",4:"answer",5:"confidence",6:"answer",7:"answer",8:"confidence",9:"answer",10:"confidence",11:"answer",12:"confidence",13:"answer",14:"confidence",15:"answer",16:"confidence"},
            "value":{0:"0.7",1:"79%",2:"0.5",3:"55%",4:"0.8",5:"88.88%",6:"0.6",7:"1",8:"100%",9:"0.85",10:"90%",11:"0.85",12:"100%",13:"0.8",14:"50%",15:"0.1111",16:"10"},
            "timestamp":{0:"2020-01-16 14:10:22",1:"2020-01-16 14:10:32",2:"2020-01-16 14:10:34",3:"2020-01-16 14:10:41",4:"2020-01-16 14:14:34",5:"2020-01-16 14:14:38",6:"2020-01-16 14:14:41",7:"2020-01-16 14:18:39",8:"2020-01-16 14:18:42",9:"2020-01-16 14:21:50",10:"2020-01-16 14:21:54",11:"2020-01-16 14:21:56",12:"2020-01-16 14:21:59",13:"2020-01-16 14:22:05",14:"2020-01-16 14:24:08",15:"2020-01-16 14:24:20",16:"2020-01-16 14:24:28"}},
            columns=['sender', 'question', 'input', 'value', 'timestamp'])
        pd_testing.assert_frame_equal(
            expected_answers, self.loader.answers)

    def test_load_influences_are_correct(self):
        expected_influences = pd.DataFrame({
            "sender":{0:"pogs10.1",1:"pogs10.2",2:"pogs10.2",3:"pogs10.1",4:"pogs10.1",5:"pogs10.2",6:"pogs10.2"},
            "question":{0:"GD_influence_surgery1",1:"GD_influence_surgery1",2:"GD_influence_surgery1",3:"GD_influence_surgery2",4:"GD_influence_surgery2",5:"GD_influence_surgery2",6:"GD_influence_surgery2"},
            "input":{0:"self",1:"self",2:"other",3:"self",4:"other",5:"self",6:"other"},
            "value":{0:"90",1:"51",2:"49",3:"1",4:"99",5:"100",6:"0"},
            "timestamp":{0:"2020-01-16 14:15:11",1:"2020-01-16 14:15:20",2:"2020-01-16 14:15:22",3:"2020-01-16 14:19:07",4:"2020-01-16 14:19:09",5:"2020-01-16 14:19:10",6:"2020-01-16 14:19:12"}},
            columns=['sender', 'question', 'input', 'value', 'timestamp'])
        pd_testing.assert_frame_equal(
            expected_influences, self.loader.influences)
    
    def test_load_frustrations_are_correct(self):
        expected_frustrations = pd.DataFrame({
            "sender":{0:"pogs10.2",1:"pogs10.1",2:"pogs10.1",3:"pogs10.2"},
            "question":{0:"GD_frustration_surgery",1:"GD_frustration_surgery",2:"GD_frustration_surgery",3:"GD_frustration_surgery"},
            "value":{0:"[\"Yes\",\"\",\"\",\"\"]",1:"[\"\",\"No\",\"\",\"\"]",2:"0",3:"5"},
            "timestamp":{0:"2020-01-16 14:22:48",1:"2020-01-16 14:22:59",2:"2020-01-16 14:23:07",3:"2020-01-16 14:23:09"}},
            columns=['sender', 'question', 'value', 'timestamp'])
        pd_testing.assert_frame_equal(
            expected_frustrations, self.loader.frustrations)

    def test_load_messages_are_correct(self):
        expected_messages = pd.DataFrame({
            "sender":{0:"pogs10.1",1:"pogs10.2",2:"pogs10.2",3:"pogs10.2",4:"pogs10.1",5:"pogs10.1",6:"pogs10.2"},
            "question":{0:"GD_group_surgery1",1:"GD_group_surgery1",2:"GD_group_surgery1",3:"GD_group_surgery1",4:"GD_group_surgery2",5:"GD_group_sports1",6:"GD_group_sports1"},
            "text":{0:"Hello there",1:"Hi!!!",2:"I have no clue",3:":)",4:"sup?",5:"bye!",6:"BYE"},
            "timestamp":{0:"2020-01-16 14:12:20",1:"2020-01-16 14:12:23",2:"2020-01-16 14:12:39",3:"2020-01-16 14:12:56",4:"2020-01-16 14:17:02",5:"2020-01-16 14:26:04",6:"2020-01-16 14:26:10"}},
            columns=['sender', 'question', 'text', 'timestamp'])
        pd_testing.assert_frame_equal(
            expected_messages, self.loader.messages)

    def test_load_answers_are_correct_in_log_with_no_confidence(self):
        expected_answers = pd.DataFrame([
            {"sender":"subj1","question":"GD_solo_disaster0","input":"answer","value":"50","timestamp":"2020-03-04 18:38:42"},
            {"sender":"subj2","question":"GD_solo_disaster0","input":"answer","value":"200","timestamp":"2020-03-04 18:38:51"},
            {"sender":"subj1","question":"GD_solo_disaster1","input":"answer","value":"55","timestamp":"2020-03-04 18:42:58"},
            {"sender":"subj2","question":"GD_solo_disaster1","input":"answer","value":"1000 mil","timestamp":"2020-03-04 18:43:02"},
            {"sender":"subj1","question":"GD_solo_disaster2","input":"answer","value":"100","timestamp":"2020-03-04 18:47:08"},
            {"sender":"subj2","question":"GD_solo_disaster2","input":"answer","value":"$88","timestamp":"2020-03-04 18:47:18"}],
            columns=['sender', 'question', 'input', 'value', 'timestamp'])
        pd_testing.assert_frame_equal(
            expected_answers, self.loader_no_confidence.answers)

    def test_load_influences_are_correct_in_log_with_no_confidence(self):
        expected_influences = pd.DataFrame([
            {"sender":"subj1","question":"GD_influence_disaster1","input":"self","value":"100","timestamp":"2020-03-04 18:43:47"},
            {"sender":"subj2","question":"GD_influence_disaster1","input":"self","value":"99","timestamp":"2020-03-04 18:43:54"},
            {"sender":"subj2","question":"GD_influence_disaster1","input":"other","value":"1","timestamp":"2020-03-04 18:43:57"},
            {"sender":"subj2","question":"GD_influence_disaster2","input":"self","value":"50","timestamp":"2020-03-04 18:47:43"},
            {"sender":"subj2","question":"GD_influence_disaster2","input":"other","value":"55","timestamp":"2020-03-04 18:47:45"},
            {"sender":"subj2","question":"GD_influence_disaster2","input":"self","value":"45","timestamp":"2020-03-04 18:47:46"}],
            columns=['sender', 'question', 'input', 'value', 'timestamp'])
        pd_testing.assert_frame_equal(
            expected_influences, self.loader_no_confidence.influences)

    
    def test_load_messages_are_correct_in_log_with_no_confidence(self):
        expected_messages = pd.DataFrame([
            {"sender":"subj1","question":"GD_group_disaster1","text":"hello","timestamp":"2020-03-04 18:40:50"},
            {"sender":"subj2","question":"GD_group_disaster1","text":"hi there","timestamp":"2020-03-04 18:40:54"},
            {"sender":"subj2","question":"GD_group_disaster1","text":"sup???","timestamp":"2020-03-04 18:41:58"},
            {"sender":"subj1","question":"GD_group_disaster2","text":"cooooooooool","timestamp":"2020-03-04 18:45:26"}],
            columns=['sender', 'question', 'text', 'timestamp'])
        pd_testing.assert_frame_equal(
            expected_messages, self.loader_no_confidence.messages)

    # =========================================================================
    # =================== get_answers_in_simple_format ========================
    # =========================================================================
    def test_get_answers_in_simple_format(self):
        expected = pd.DataFrame({
            "Question":{0:"sports0",1:"surgery0",2:"surgery1",3:"surgery2",4:"surgery3"},
            "pogs10.1's answer":{0:"",1:"0.7",2:"0.8",3:"",4:"0.85"},
            "pogs10.1's confidence":{0:"50%",1:"79%",2:"88.88%",3:"",4:"90%"},
            "pogs10.2's answer":{0:"0.1111",1:"0.5",2:"0.6",3:"1",4:"0.8"},
            "pogs10.2's confidence":{0:"10",1:"55%",2:"",3:"100%",4:"100%"}})
        computed = self.loader.get_answers_in_simple_format()
        pd_testing.assert_frame_equal(expected, computed)

    # =========================================================================
    # ===================== get_influence_matrices2x2 ==========================
    # =========================================================================
    def test_get_influence_matrices2x2(self):
        expected_question_orders = ['surgery1', 'surgery2']
        expected_influence_matrices = [
            np.array([[0.9, 0.1],
                      [0.49, 0.51]]),
            np.array([[0.01, 0.99],
                      [0.0, 1.0]])]
        expected_influences_from_data = [
            np.array([[True, False], [True, True]]),
            np.array([[True, True], [True, True]])
        ]
        computed_questions_order, computed_influence_matrices, computed_influences_from_data = (
            self.loader.get_influence_matrices2x2(make_it_row_stochastic=True))
        np_testing.assert_array_equal(
            expected_question_orders, computed_questions_order)
        np_testing.assert_array_equal(
            expected_influence_matrices,
            computed_influence_matrices)
        np_testing.assert_array_equal(
            expected_influences_from_data,
            computed_influences_from_data)

    @parameterized.expand([
        ['with_one_missing',
            pd.DataFrame({
                "sender":{"0":"pogs10.1","1":"pogs10.2","2":"pogs10.2"},
                "question":{"0":"GD_influence_surgery1","1":"GD_influence_surgery1","2":"GD_influence_surgery1"},
                "input":{"0":"self","1":"self","2":"other"},"value":{"0":"90","1":"51","2":"49"},
                "timestamp":{"0":"2020-01-16 14:15:11","1":"2020-01-16 14:15:20","2":"2020-01-16 14:15:22"}}),
        [np.array([[0.9, 0.1],
                   [0.49, 0.51]])],
        ],
        ['with_one_missing_and_one_empty',
            pd.DataFrame({
                "sender":{"0":"pogs10.1","1":"pogs10.2","2":"pogs10.2"},
                "question":{"0":"GD_influence_surgery1","1":"GD_influence_surgery1","2":"GD_influence_surgery1"},
                "input":{"0":"self","1":"self","2":"other"},"value":{"0":"","1":"51","2":"49"},
                "timestamp":{"0":"2020-01-16 14:15:11","1":"2020-01-16 14:15:20","2":"2020-01-16 14:15:22"}}),
        [np.array([[0.5, 0.5],
                   [0.49, 0.51]])],
        ],
        ['with_only_one',
            pd.DataFrame({
                "sender":{"0":"pogs10.1"},
                "question":{"0":"GD_influence_surgery1"},
                "input":{"0":"self"},"value":{"0":"50",},
                "timestamp":{"0":"2020-01-16 14:15:11"}}),
        [np.array([[0.50, 0.50],
                   [0.50, 0.50]])],
        ],
        ['with_larger_values',
            pd.DataFrame({
                "sender":{"0":"pogs10.1","1":"pogs10.2","2":"pogs10.2"},
                "question":{"0":"GD_influence_surgery1","1":"GD_influence_surgery1","2":"GD_influence_surgery1"},
                "input":{"0":"self","1":"self","2":"other"},"value":{"0":"","1":"60","2":"90"},
                "timestamp":{"0":"2020-01-16 14:15:11","1":"2020-01-16 14:15:20","2":"2020-01-16 14:15:22"}}),
        [np.array([[0.5, 0.5],
                   [0.6, 0.4]])],
        ],
        ['with_duplicate_due_to_change_of_value',
            pd.DataFrame({
                "sender":{"0":"pogs10.1","0":"pogs10.1","1":"pogs10.2","2":"pogs10.2"},
                "question":{"0":"GD_influence_surgery1","0":"GD_influence_surgery1","1":"GD_influence_surgery1","2":"GD_influence_surgery1"},
                "input":{"0":"self","0":"self","1":"self","2":"other"},
                "value":{"0":"55","0":"5","1":"51","2":"49"},
                "timestamp":{"0":"2020-01-16 14:15:11","0":"2020-01-16 14:15:12","1":"2020-01-16 14:15:20","2":"2020-01-16 14:15:22"}}),
        [np.array([[0.05, 0.95],
                   [0.49, 0.51]])],
        ]])
    def test_get_influence_matrices2x2_mocked(self, name, influences, expected_influence_matrices):
        self.loader.influences = influences
        _, computed_influence_matrices, _ = (
            self.loader.get_influence_matrices2x2(make_it_row_stochastic=True))
        np_testing.assert_array_equal(
            expected_influence_matrices,
            computed_influence_matrices)

    # =========================================================================
    # ============== get_frustrations_in_simple_format ========================
    # =========================================================================
    def test_get_frustrations_in_simple_format(self):
        expected = pd.DataFrame({
            "Question":{0: "surgery"},
            "pogs10.1's answer":{0: "0"},
            "pogs10.2's answer":{0: "5"}})
        computed = self.loader.get_frustrations_in_simple_format()
        pd_testing.assert_frame_equal(expected, computed)

    # =========================================================================
    # =============== get_all_groups_info_in_one_dataframe ====================
    # =========================================================================
    def test_get_all_groups_info_in_one_dataframe(self):
        teams_log_list = {'s10': self.loader}
        dt = [
            ['s10', '1', 'asbestos', '', '', '', '', '', '', '', '', '', ''],
            ['s10', '1', 'disaster', '', '', '', '', '', '', '', '', '', ''],
            ['s10', '1', 'sports', '', '', '', '', '', '', '', '', '', ''],
            ['s10', '1', 'school', '', '', '', '', '', '', '', '', '', ''],
            ['s10', '1', 'surgery', '0.7', '0.8', '0.9', '0.1', '', '0.01', '0.99', '0.85', '', ''],
            ['s10', '2', 'asbestos', '', '', '', '', '', '', '', '', '', ''],
            ['s10', '2', 'disaster', '', '', '', '', '', '', '', '', '', ''],
            ['s10', '2', 'sports', '0.1111', '', '', '', '', '', '', '', '', ''],
            ['s10', '2', 'school', '', '', '', '', '', '', '', '', '', ''],
            ['s10', '2', 'surgery', '0.5', '0.6', '0.51', '0.49', '1', '1.0', '0.0', '0.8', '', '']]
        expected = pd.DataFrame(dt, columns = [
            'Group', 'Person', 'Issue', 'Initial opinion',
            'Period1 opinion', 'Period1 wii', 'Period1 wij',
            'Period2 opinion', 'Period2 wii', 'Period2 wij',
            'Period3 opinion', 'Period3 wii', 'Period3 wij'])
        computed = gll.get_all_groups_info_in_one_dataframe(
            teams_log_list)
        pd_testing.assert_frame_equal(expected, computed)

    # =========================================================================
    # ============== compute_attachment_to_initial_opinion ====================
    # =========================================================================
    def test_compute_attachment_raises_when_not_matching_opinions(self):
        x1 = [0.1, 0.2, 0.6, 0.4]
        x2 = [0.9, 0.4, 0.7]
        w12 = [0.1, 0.0, 0.2]
        with self.assertRaises(ValueError):
            gll.compute_attachment_to_initial_opinion(x1, x2, w12)

    def test_compute_attachment_raises_when_not_matching_opinions_influence(
            self):
        x1 = [0.1, 0.2, 0.6, 0.4]
        x2 = [0.9, 0.4, 0.7, 0.5]
        w12 = [0.1, 0.0]
        with self.assertRaises(ValueError):
            gll.compute_attachment_to_initial_opinion(x1, x2, w12)

    def test_compute_attachment_to_initial_opinion(self):
        x1 = [0.1, 0.2, 0.6, 0.4]
        x2 = [0.9, 0.4, 0.7, 0.5]
        w12 = [0.1, 0.0, 0.2]
        expected_a11 = [0.1/0.08, 0.5/0.1, 0.3/0.52]
        computed_a11 = gll.compute_attachment_to_initial_opinion(
            xi=x1, xj=x2, wij=w12, eps=0)
        np_testing.assert_array_almost_equal(expected_a11, computed_a11)

    def test_compute_attachment_to_initial_opinion_when_division_by_zero(self):
        x1 = [0.2, 0.2, 0.2]
        x2 = [0.4, 0.4, 0.4]
        w12 = [0.1, 0.0]
        expected_a11 = [0 / 0.02,
                        np.nan]
        computed_a11 = gll.compute_attachment_to_initial_opinion(
            xi=x1, xj=x2, wij=w12, eps=0)
        np_testing.assert_array_almost_equal(expected_a11, computed_a11)

    def test_compute_attachment_to_initial_when_division_by_zero_with_eps(self):
        x1 = [0.2, 0.2, 0.2]
        x2 = [0.4, 0.4, 0.4]
        w12 = [0.1, 0.0]
        eps = 0.01
        expected_a11 = [(0 + eps) / (0.02 + eps),
                        (0 + eps) / (0 + eps)]
        computed_a11 = gll.compute_attachment_to_initial_opinion(
            xi=x1, xj=x2, wij=w12, eps=eps)
        np_testing.assert_array_almost_equal(expected_a11, computed_a11)

    # =========================================================================
    # ================== compute_all_teams_attachments ========================
    # =========================================================================
    def test_compute_all_teams_attachments(self):
        teams_data = {
            55: {
                'asbestos': {
                    'w12': [0.1, 0.0, 0.2],
                    'w21': [0.0, 0.0, 0.0],
                    'x1': [0.1, 0.2, 0.6, 0.4],
                    'x2': [0.9, 0.4, 0.7, 0.5]},
                'surgery': {
                    'w12': [0.35, 0.4, 0.5],
                    'w21': [0.25, 0.3, 0.3],
                    'x1': [0.6, 0.65, 0.7, 0.7],
                    'x2': [0.75, 0.5, 0.6, 0.7]}}}
        expected_attachments = {
            55: {
                'asbestos': {
                    'a11': [0.1/0.08, 0.5/0.1, 0.3/0.52],
                    'a22': [np.nan, -0.2/-0.5, -0.4/-0.2]},  # Nan was -0.5/0.
                'surgery': {
                    'a11': [0.05/(0.35*0.15),
                            0.1/(0.05+0.4*-0.15),
                            0.1/(0.1+0.5*-0.1)],
                    'a22': [-0.25/(0.25*-0.15),
                            -0.15/(-0.25+0.3*0.15),
                            -0.05/(-0.15+0.3*0.1)]}}}
        computed_attachments = gll.compute_all_teams_attachments(
            teams_data=teams_data, eps=0)
        utils.assert_dict_equals(
            d1=expected_attachments,
            d2=computed_attachments,
            almost_number_of_decimals=6)
