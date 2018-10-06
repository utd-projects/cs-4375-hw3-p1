# import sys


class MDP(object):
    """Markov decision process implementation.

    Attributes
    ----------
    discount: float
        The discount factor for moving to another state.
    states: dict of MDP.state
        The states in the MDP.

    Methods
    -------
    __init__(discount: float, input_file_name: str = None)
        MDP initializer

    parse_file(input_file_name)
        Parse file to initilize the MDP's states.

    See Also
    --------
    MDP.state
    """

    class state(object):
        """State instance found in Markov Decision Process.

        Attributes
        ----------
        reward: int or float
            The reward associated for going to this specific state.
        actions: dict of dict of float
            The actions the state and take and their corresponding
            probabilities for happening.

            Useage:
                self.actions[action] = name of states that it can go to.
                self.actions[action][to_state] = chance of going to to_state.
        """

        def __init__(self, reward: int, actions_arr: list):
            self.reward = reward
            self.actions = MDP.state._parse_line(actions_arr)

        def _parse_line(actions_arr: list) -> dict:
            actions = dict()

            for i in range(0, len(actions_arr), 3):
                action = actions_arr[i][1:]
                to_state = actions_arr[i+1]
                probability = float(actions_arr[i+2][:-1])

                if action not in actions:
                    actions[action] = dict()
                actions[action][to_state] = probability

            return actions

    def __init__(self, discount: float, input_file_name: str = None):
        """MDP initializer

        Parameters
        ----------
        discount: float
            The discount factor for moving to another state.
        input_file_name: str, optional
            The name of the input file that holds the the MDP's state
            information.

        See Also
        --------
        parse_file
        """

        self.discount = discount
        self.states = dict()
        if input_file_name:
            self.parse_file(input_file_name)

        self.optimal_policies = [
            {key: dict() for key in self.states.keys()}
        ]
        for state in self.states.keys():
            self.optimal_policies[0][state]["action"] = None
            self.optimal_policies[0][state]["j_val"] = \
                self.states[state].reward

    def parse_file(self, input_file_name):
        """Parse file to initilize the MDP's states.

        A single valid state information line has the following:

        state_name reward (action_1 state1 prob1) (action_2 state2 prob 2) ...

        Parameters
        ----------
        input_file_name: str
            The name of the input file that holds the the MDP's state
            information.
        """

        self.states = dict()
        with open(input_file_name, 'r') as file:
            for line in file:
                if not line.isspace():
                    line_arr = line.split()
                    self.states[line_arr[0]] = MDP.state(int(line_arr[1]),
                                                         line_arr[2:])

    def find_optimal_policies(self, iterations: int = 20):
        """Find the optimal policies up to the requested number of iterations.

        Parameters
        ----------
        iterations: int, optional
            The maximum number of iterations that the optimal policy will be
            calculated for. Default value is 20.
        """

        if len(self.optimal_policies) < iterations:
            for i in range(len(self.optimal_policies), iterations):
                self.optimal_policies.append(dict())
                for state in self.states:
                    self.optimal_policies[i][state] = dict()
                    best_action = self._get_best_action(state, i)
                    self.optimal_policies[i][state]["action"] = best_action[0]
                    self.optimal_policies[i][state]["j_val"] = \
                        self.states[state].reward + \
                        (self.discount * best_action[1])

    def _get_best_action(self, state: str, iteration: int) -> list:
        """Gets the best action  based on the current state and iteration.

        Parameters
        ----------
        state: str
            The state that the method finds the best policy for.
        iteration: int
            The iteration point that bases which values to use to determine
            the best action.

        Returns
        -------
        best: [str, float]
            The name of the best action and its corresponding value based on
            taking that action.
        """

        curr_best = ["", float("-inf")]

        for action, action_results in self.states[state].actions.items():
            temp_max = float(0)
            for to_state, probability in action_results.items():
                temp_max += probability * \
                    self.optimal_policies[iteration-1][to_state]["j_val"]

            if curr_best[1] < temp_max:
                curr_best[1] = temp_max
                curr_best[0] = action

        return curr_best

    def __str__(self):
        """String representation of all calculated optimal policy iterations

        See Also
        --------
        optimal_policy_strs
        """

        return self.optimal_policy_strs(len(self.optimal_policies))

    def optimal_policy_strs(self, end: int):
        """Get the states' optimal policy from interation 1 to given end point

        Parameters
        ----------
        end: int
            The last iteration to get. If the value is greater than the number
            of stored iterations, then the last calculated policy iterations
            are used instead.

        Returns
        -------
        str_iterations: str
            Single string that shows all the states' optimal policies from
            iteration 1 to the given end point. Iteration instances are on
            their own line.
        """

        return '\n'.join([self._optimal_policy_str(i) for i in
                          range(min(end, len(self.optimal_policies)))])

    def _optimal_policy_str(self, iteration: int):
        """Get the states' optimal policy at the given iteration point

        Parameters
        ----------
        iteration: int
            The iteration point to retrieve the states' optimal policy
            information

        Returns
        -------
        interation_str: str
            String representation of the states' optimal policy at the given
            iteration point.
        """

        iteration_str = "After iteration " + str(iteration+1) + ":"
        for state, state_results in self.optimal_policies[iteration].items():
            iteration_str += ' (' + state + ' ' + \
                     str(state_results["action"]) + ' ' + \
                     format(state_results["j_val"], ".4f") + ')'
        return iteration_str


def main():
    mdp = MDP(0.9, "test.in")
    mdp.find_optimal_policies()
    print(str(mdp))


if __name__ == "__main__":
    main()
