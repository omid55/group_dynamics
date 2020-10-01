###############################################################################
# Omid55
# Start date:     16 Jan 2020
# Modified date:  14 Apr 2020
# Author:   Omid Askarisichani
# Email:    omid55@cs.ucsb.edu
# Module to load group dynamics logs for every team.
###############################################################################

from __future__ import division, print_function, absolute_import, unicode_literals

import glob
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from enum import Enum, unique
from os.path import expanduser
from typing import Dict
from typing import List
from typing import Text
from typing import Tuple

import utils


@unique
class AttachmentType(Enum):
    TO_INITIAL = 1
    TO_PREVIOUS = 2


class TeamLogsLoader(object):
    """Processes the logs of one team who have finished the dyad group dynamics

        Usage:
            loader = TeamLogsLoader(
                directory='/home/omid/Datasets/Jeopardy')

        Properties:
            team_id: The id of the existing team in this object.

    """
    taskid2taskname = {
        52: 'GD_solo_asbestos_initial',
        53: 'GD_group_asbestos1',
        57: 'GD_influence_asbestos1',
        61: 'GD_group_asbestos2',
        62: 'GD_solo_asbestos1',
        63: 'GD_influence_asbestos2',
        67: 'GD_solo_asbestos2',
        68: 'GD_group_asbestos3',
        69: 'GD_influence_asbestos3',
        70: 'GD_solo_asbestos3',
        71: 'GD_solo_disaster_initial',
        72: 'GD_group_disaster1',
        73: 'GD_solo_disaster1',
        74: 'GD_influence_disaster1',
        75: 'GD_solo_disaster3',
        76: 'GD_solo_disaster2',
        77: 'GD_influence_disaster3',
        78: 'GD_influence_disaster2',
        79: 'GD_group_disaster2',
        80: 'GD_group_disaster3',
        81: 'GD_frustration_asbestos',
        82: 'GD_frustration_disaster',
        83: 'GD_solo_sports_initial',
        84: 'GD_solo_sports1',
        85: 'GD_solo_sports2',
        86: 'GD_solo_sports3',
        87: 'GD_group_sports2',
        88: 'GD_group_sports3',
        89: 'GD_group_sports1',
        90: 'GD_influence_sports3',
        91: 'GD_influence_sports2',
        92: 'GD_influence_sports1',
        93: 'GD_frustration_sports',
        94: 'GD_solo_school_initial',
        95: 'GD_solo_school1',
        96: 'GD_solo_school2',
        97: 'GD_solo_school3',
        98: 'GD_group_school3',
        99: 'GD_group_school2',
        100: 'GD_group_school1',
        101: 'GD_frustration_school',
        102: 'GD_influence_school3',
        103: 'GD_influence_school2',
        104: 'GD_influence_school1',
        105: 'GD_solo_surgery_initial',
        106: 'GD_solo_surgery1',
        107: 'GD_solo_surgery2',
        108: 'GD_solo_surgery3',
        109: 'GD_group_surgery1',
        110: 'GD_group_surgery2',
        111: 'GD_group_surgery3',
        112: 'GD_influence_surgery1',
        113: 'GD_influence_surgery2',
        114: 'GD_influence_surgery3',
        115: 'GD_frustration_surgery'}
    
    def __init__(self, directory: Text):
        self.messages = None
        self.answers = None
        self.influences = None
        self.frustrations = None
        self._load(directory=directory)

    def _load(self, directory: Text):
        """Loads all logs for one team in the given directory.
        """
        logs_filepath = '{}/EventLog_*.csv'.format(directory)
        logs = pd.read_csv(glob.glob(expanduser(logs_filepath))[0])
        logs = logs.sort_values(by='Timestamp')

        task_orders_filepath = '{}/CompletedTask_*.csv'.format(directory)
        task_orders = pd.read_csv(
            glob.glob(expanduser(task_orders_filepath))[0])

        completed_taskid2taskname = {}
        for index, row in task_orders.iterrows():
            completed_taskid2taskname[row.Id] = TeamLogsLoader.taskid2taskname[
                row.TaskId]

        answers_dt = []
        messages_dt = []
        influences_dt = []
        frustrations_dt = []

        for index, row in logs.iterrows():
            content_file_id = row.EventContent[9:]
            question_name = completed_taskid2taskname[row.CompletedTaskId]
            sender = row.Sender
            timestamp = row.Timestamp
            event_type = row.EventType    
            json_file_path = '{}/EventLog/{}_*.json'.format(
                directory, content_file_id)
            json_file = glob.glob(expanduser(json_file_path))
            if len(json_file) != 1:
                print(
                    'WARNING1: json file for id: {} was not found'
                    ' in the EventLog folder.\n'.format(
                        content_file_id))
            else:
                with open(json_file[0], 'r') as f:
                    content = json.load(f)
                    if 'type' in content and content['type'] == 'JOINED':
                        continue
                    if event_type == 'TASK_ATTRIBUTE':
                        input_str = ''
                        if question_name[:7] == 'GD_solo':
                            if content['attributeName'] == 'surveyAnswer0':
                                input_str = 'answer'
                            elif content['attributeName'] == 'surveyAnswer1':
                                input_str = 'confidence'
                            else:
                                print('WARNING2: attribute name was unexpected.'
                                    ' It was {}, question was {} '
                                    'and content was {}\n'.format(
                                        content['attributeName'],
                                        question_name,
                                        content))
                            if question_name.endswith('_initial'):
                                question_name = question_name.split(
                                    '_initial')[0] + '0'
                            answers_dt.append(
                                [sender, question_name, input_str,
                                content['attributeStringValue'], timestamp])
                        elif question_name[:12] == 'GD_influence':
                            if content['attributeName'] == 'surveyAnswer1':
                                input_str = 'self'
                            elif content['attributeName'] == 'surveyAnswer2':
                                input_str = 'other'
                            else:
                                print('WARNING3: attribute name was unexpected.'
                                ' It was {}\n'.format(content['attributeName']))
                            influences_dt.append(
                                [sender, question_name, input_str,
                                content['attributeStringValue'], timestamp])
                        elif question_name[:14] == 'GD_frustration':
                            frustrations_dt.append(
                                [sender, question_name,
                                content['attributeStringValue'], timestamp])
                        else:
                            print('WARNING4: There was an unexpected '
                            'question: {}\n'.format(question_name))
                    elif event_type == 'COMMUNICATION_MESSAGE':
                        if len(content['message']) > 0:
                            messages_dt.append(
                                [sender, question_name,
                                content['message'], timestamp])
                    else:
                        print('WARNING5: There was an unexpected'
                        ' EventType: {}\n'.format(event_type))

        self.answers = pd.DataFrame(answers_dt, columns = [
            'sender', 'question', 'input', 'value', 'timestamp'])
        self.influences = pd.DataFrame(influences_dt, columns = [
            'sender', 'question', 'input', 'value', 'timestamp'])
        self.frustrations = pd.DataFrame(frustrations_dt, columns = [
            'sender', 'question', 'value', 'timestamp'])
        self.messages = pd.DataFrame(messages_dt, columns = [
            'sender', 'question', 'text', 'timestamp'])
        # Sorts all based on timestamp.
        self.answers.sort_values(by='timestamp', inplace=True)
        self.influences.sort_values(by='timestamp', inplace=True)
        self.frustrations.sort_values(by='timestamp', inplace=True)
        self.messages.sort_values(by='timestamp', inplace=True)
        self.users = np.unique(
            self.influences.sender.tolist() +
            self.messages.sender.tolist() +
            self.answers.sender.tolist())
        if self.users.size > 0:
            self.team_id = self.users[0].split('.')[0]

    def get_answers_in_simple_format(self) -> pd.DataFrame:
        """Gets all answers in a simple format to read them in the easiest way.
        """
        # if len(self.answers) == 0:
        #     raise ValueError('The object has not been initialized.')
        users = self.users
        questions = np.unique(self.answers.question)
        data = []
        for question in questions:
            dt = [question[len('GD_solo_'):]]
            for user in users:
                for input in ['answer', 'confidence']:
                    this_answer = self.answers[
                        (self.answers.question == question) & (
                            self.answers.sender == user) & (
                                self.answers.input == input)]
                    val = ''
                    if len(this_answer.value) > 0:
                        # Because if there might be multiple log entry for the
                        #  same text box, we take the last one.
                        val = list(this_answer.value)[-1]
                    dt.append(val)
            data.append(dt)
        columns = ['Question']
        for user in users:
            columns += [user + '\'s answer', user + '\'s confidence']
        return pd.DataFrame(data, columns=columns)

    def get_influence_matrices2x2(
            self,
            make_it_row_stochastic: bool = True
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gets influence matrices in 2 * 2 format.
        
        If empty or missing string, it fills with 100 - other one. If both
        empty or missing it fills both with 50.
        """
        influence_matrices = []
        influences_from_data = []
        users = self.users
        questions = np.unique(self.influences.question)
        for question in questions:
            influences = []
            for user in users:
                for input in ['self', 'other']:
                    this_influence = self.influences[
                        (self.influences.question == question) & (
                            self.influences.sender == user) & (
                                self.influences.input == input)]
                    val = ''
                    if len(this_influence.value) > 0:
                        # Because if there might be multiple log entry for the
                        #  same text box, we take the last one.
                        val = list(this_influence.value)[-1]
                    val = str(val).split('%')[0]
                    influences.append(val)
            tmp = influences[2]
            influences[2] = influences[3]
            influences[3] = tmp
            influences = np.reshape(influences, (2, 2))
            empty_strings = np.where(influences == '')
            influence_from_data = np.ones((2, 2), dtype=np.bool)
            for l in range(len(empty_strings[0])):
                i = empty_strings[0][l]
                j = empty_strings[1][l]
                if influences[i, 1-j] == '':
                    influences[i, 1-j] = 50
                    influence_from_data[i, 1-j] = False
                influences[i, j] = 100 - float(influences[i, 1-j])
                influence_from_data[i, j] = False
            influences = np.array(influences, dtype=np.float)
            if make_it_row_stochastic:
                influences = utils.make_matrix_row_stochastic(influences)
            influence_matrices.append(influences)
            influences_from_data.append(influence_from_data)
        question_names = [
            question[len('GD_influence_'):] for question in questions]
        return question_names, np.array(influence_matrices), np.array(
            influences_from_data)

    def get_frustrations_in_simple_format(self):
        """Gets all frustrations in a simple format to be read."""
        users = self.users
        questions = np.unique(self.frustrations.question)
        data = []
        for question in questions:
            dt = [question[len('GD_frustration_'):]]
            for user in users:
                this_frustration = self.frustrations[(
                    self.frustrations.question == question) & (
                        self.frustrations.sender == user)]
                val = ''
                if len(this_frustration.value) > 0:
                    # Because if there might be multiple log entry for the
                    #  same text box, we take the last one.
                    vals = []
                    for v in list(this_frustration.value):
                        if v[0] != '[':
                            vals.append(v)
                    if len(vals) > 0:
                        val = vals[-1]
                dt.append(val)
            data.append(dt)
        columns = ['Question']
        for user in users:
            columns += [user + '\'s answer']
        return pd.DataFrame(data, columns=columns)

    def get_combined_messages(self) -> pd.DataFrame:
        pass


def get_all_groups_info_in_one_dataframe(
        teams_log: Dict[Text, TeamLogsLoader]) -> pd.DataFrame:
    """Gets all teams' logs in one dataframe.
    """
    dt = []
    issues = ['asbestos', 'disaster', 'sports', 'school', 'surgery']
    for team_log in teams_log.values():
        answers = team_log.get_answers_in_simple_format()
        q_order, inf_matrices , from_data = team_log.get_influence_matrices2x2(
            make_it_row_stochastic=True)
        for index, user in enumerate(team_log.users):
            user_ans = answers[['Question', user + '\'s answer']]
            # user_conf = answers[['Question', user + '\'s confidence']]
            for issue in issues:
                vals = []
                for i in range(4):
                    op = user_ans[user_ans['Question'] == issue + str(i)]
                    # co = user_ans[user_conf['Question'] == issue + str(i)]

                    op_v = ''
                    if len(op.values) > 0:
                        op_v = op.values[0][1]
                    if issue == 'asbestos' or issue == 'disaster':
                        if len(op_v) > 0:
                            op_v = op_v.replace('$', '')
                            op_v = op_v.replace('dollars', '')
                            op_v = op_v.replace('per person', '')
                            op_v = op_v.replace(',', '')
                            op_v = op_v.replace('.000', '000')
                            op_v = op_v.replace(' million', '000000')
                            op_v = op_v.replace(' mil', '000000')
                            op_v = op_v.replace(' M', '000000')
                            op_v = op_v.replace('M', '000000')
                            op_v = op_v.replace(' k', '000')
                            op_v = op_v.replace('k', '000')
                            op_v = op_v.replace(' K', '000')
                            op_v = op_v.replace('K', '000')
                            op_v = '$' + op_v
                    else:
                        if len(op_v) > 0:
                            op_v = op_v.replace('%', '')
                            if op_v.isdigit() and float(op_v) > 2:
                                op_v = str(float(op_v) / 100)
                            if '/' in op_v:
                                nums = op_v.split('/')
                                op_v = str(int(nums[0]) / int(nums[1]))
                    vals.append(op_v)

                    if i > 0:
                        wii = ''
                        wij = ''
                        v = np.where(np.array(q_order) == issue + str(i))[0]
                        if len(v) > 0:
                            if from_data[v[0]][index, index] or from_data[v[0]][index, 1 - index]:
                                wii = round(inf_matrices[v[0]][index, index], 2)
                                wij = round(inf_matrices[v[0]][index, 1 - index], 2)
                                wii = str(wii)
                                wij = str(wij)
                        vals.extend([wii, wij])
                dt.append(
                    [team_log.team_id[3:], user.split('.')[1], issue] + vals)
    data = pd.DataFrame(dt, columns = [
        'Group', 'Person', 'Issue', 'Initial opinion',
        'Period1 opinion', 'Period1 wii', 'Period1 wij',
        'Period2 opinion', 'Period2 wii', 'Period2 wij',
        'Period3 opinion', 'Period3 wii', 'Period3 wij'])
    return data


def compute_attachment(
        xi: List[float],
        xj: List[float],
        wij: List[float],
        start_k: int = 0,
        eps: float = 0.0,
        to_opinion: AttachmentType = AttachmentType.TO_INITIAL
        ) -> Tuple[List[float], List[Dict[Text, int]]]:
    """Computes the level of attachment to the initial opinion for person i.

    For person i at time k is computed as follows:
    a_{i, i}(k) = \frac{x_i(k+1) - x_i(0) + \epsilon}{x_i(k) - x_i(0) + w_{i, j}(k)\bigg(x_j(k) - x_i(k)\bigg) + \epsilon}

    Args:
        xi: List of opinions for person i over time (multiple rounds).

        xj: List of opinions for person j over time (multiple rounds).

        wij: List of influence from person i to person j.

        start_k: Start value for k to be 0 or 1.

        eps: The small amount to always add to the denominator.

        to_opinion: The type of attachment to either initial or previous.

    Returns:
        Value (level) of attachment to the initial opinion or from previous for
        person i over time, called a_{i, i}. Also note if you want to avoid the
        possibility of division by zero, you should always use a small epsilon
        larger than zero to add to the denominator.

    Raises:
        ValueError: If the length of opinions were not the same or number of 
        influence weights were not one less than the length of opinion vector.
        Also if start_k was given anything but 0 or 1. Also if the type of 
        to_opinion was unkown.
    """
    if len(xi) != len(xj):
        raise ValueError(
            'Length of opinions do not match. xi: {}, xj: {}'.format(xi, xj))
    if len(xi) != len(wij) + 1:
        raise ValueError(
            'Length of opinions and influences do not match. ')
    if start_k != 0 and start_k != 1:
        raise ValueError('Start k should be 0 or 1. It was given {}'.format(
            start_k))
    opinion_len = len(xi)
    aii_nan_details = []
    aii = []
    for k in range(start_k, opinion_len - 1):
        if to_opinion == AttachmentType.TO_INITIAL:
            xi_k_minus_x0_or_previous_str = 'xi[k]-xi[0]==0'
            numerator = xi[k+1] - xi[0]
            denominator = xi[k] - xi[0] + wij[k] * (xj[k] - xi[k]) + eps
        elif to_opinion == AttachmentType.TO_PREVIOUS:
            xi_k_minus_x0_or_previous_str = 'xi[k]-xi[k-1]==0'
            if k > 0:
                numerator = xi[k+1] - xi[k-1]
                denominator = xi[k] - xi[k-1] + wij[k] * (xj[k] - xi[k]) + eps
            else:
                numerator = xi[k+1] - xi[k]
                denominator = wij[k] * (xj[k] - xi[k]) + eps
        else:
            ValueError('Type of attachment was unkown. It was {}'.format(
                to_opinion))
        # Getting the details about how NaN is happening.
        aii_nan_detail = {}
        if utils.is_almost_zero(numerator) and utils.is_almost_zero(denominator):
            aii_nan_detail['0/0'] = 1
        if not utils.is_almost_zero(numerator) and utils.is_almost_zero(denominator):
            aii_nan_detail['n/0'] = 1
        if utils.is_almost_zero(denominator) and utils.is_almost_zero(xi[k] - xi[0]):
            aii_nan_detail[xi_k_minus_x0_or_previous_str] = 1
        if utils.is_almost_zero(denominator) and utils.is_almost_zero(wij[k]):
            aii_nan_detail['wij[k]==0'] = 1
        if utils.is_almost_zero(denominator) and utils.is_almost_zero(xj[k] - xi[k]):
            aii_nan_detail['xj[k]-xi[k]==0'] = 1
        if utils.is_almost_zero(denominator) and eps > 0:
            print('Warning: choose a different epsilon.'
                  ' There has been an denominator equals 0'
                  ' with the current one which is {}.'.format(eps))
        # attachment = 0  #  np.nan  << CHECK HERE >> DUE TO NOAH'S CODE IS SET 0.
        attachment = np.nan
        if not utils.is_almost_zero(denominator):
            attachment = numerator / denominator
        aii.append(attachment)
        aii_nan_details.append(aii_nan_detail)
    return aii, aii_nan_details


def compute_all_teams_attachments(
        teams_data: Dict[Text, Dict[Text, Dict[Text, List[float]]]],
        start_k: int = 0,
        use_attachment_to_initial_opinion: bool = True,
        eps: float = 0.0) -> Dict[Text, Dict[Text, Dict[Text, List[float]]]]:
    """Computes all of teams' attachments.

    Args:
        teams_data: Dictionary of opinions and inf. weights for dyads over time.

        start_k: Start value for k to be 0 or 1.

        use_attachment_to_initial_opinion: If true using attachment to the initial opinion.

        eps: The small amount to always add to both numerator and denominator.

    Returns:
        Dictionary of teams with their attachment to the intial opinion.

    Raises:
        ValueError: If function compute_attachment_to_initial_opinion raises
        due to incorrect size of opinions, influence weights, or start k value.
    """
    teams_attachment = defaultdict(dict)
    for _, team_id in enumerate(teams_data.keys()):
        for _, issue in enumerate(teams_data[team_id].keys()):
            # Computing the level of attachment to the initial opinion (aii).
            this_team_issue = teams_data[team_id][issue]
            if use_attachment_to_initial_opinion:
                to_opinion = AttachmentType.TO_INITIAL
            else:
                to_opinion = AttachmentType.TO_PREVIOUS
            a11, a11_nan_details = compute_attachment(
                xi=this_team_issue['x1'],
                xj=this_team_issue['x2'],
                wij=this_team_issue['w12'],
                start_k=start_k,
                eps=eps,
                to_opinion=to_opinion)
            a22, a22_nan_details = compute_attachment(
                xi=this_team_issue['x2'],
                xj=this_team_issue['x1'],
                wij=this_team_issue['w21'],
                start_k=start_k,
                eps=eps,
                to_opinion=to_opinion)
            teams_attachment[team_id][issue] = {
                'a11': a11,
                'a22': a22,
                'a11_nan_details': a11_nan_details,
                'a22_nan_details': a22_nan_details}
    return teams_attachment


def predict_FJ(A, W, x, x0=None):
    n, _ = W.shape
    if x0 is None:
        x0 = x
    for _ in range(100):
#         x = np.matmul(np.matmul(A, W), x) + np.matmul((np.eye(n) - A), x0)
        x = A @ W @ x + (np.eye(n) - A) @ x0
    return x


def predict_Degroot(W, x):
    for _ in range(100):
#         x = np.matmul(W, x)
        x = W @ x
    return x
