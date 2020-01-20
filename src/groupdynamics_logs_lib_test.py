# Omid55
# Test module for group dynamics logs library.


from __future__ import division, print_function, absolute_import, unicode_literals

import os
import unittest
import numpy as np
import pandas as pd
from pandas import testing as pd_testing
from numpy import testing as np_testing
import groupdynamics_logs_lib


class TestTeamLogsLoaderLoad(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.loader = groupdynamics_logs_lib.TeamLogsLoader(
            directory=os.getcwd() + '/src/testing_log')

    @classmethod
    def tearDown(cls):
        del cls.loader

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
    # ======================= get_influence_matrices ==========================
    # =========================================================================
    def test_get_influence_matrices(self):
        expected_question_orders = [
            'surgery1', 'surgery2']
        expected_influence_matrices = [np.array(
            [['90', ''], ['49', '51']]), np.array([['1', '99'], ['0', '100']])]
        computed_questions_order, computed_influence_matrices = (
            self.loader.get_influence_matrices())
        np_testing.assert_array_equal(
            expected_question_orders, computed_questions_order)
        np_testing.assert_array_equal(
            expected_influence_matrices, computed_influence_matrices)
