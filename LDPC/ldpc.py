"""
Single-user LLR-domain binary LDPC decoder.

Classes:

    CheckNode
    VariableNode
    Decoder

Functions:

    Decoder.reset()
    Decoder.decodeRxSequence()

Variables:

    Decoder.parameters

"""

__author__ = "Jamison Ebert" 
__version__ = 1.1

import numpy as np


class CheckNode():
    """
    Class for single check node that implements LLR-domain message passing.

        Attributes:
            None

        Methods:
            add_edge(edge, neighbor): add edge to check node
            send_messages(messages): pass messages to adjacent variable nodes
            check_consistency(estimate): check for local parity consistency 
    """

    def __init__(self, num_edges):
        """
        Initialize CheckNode class. 

            Parameters:
                num_edges (int): number of connected edges

            Returns: 
                <none>
        """
        self.__edges = np.zeros(num_edges, dtype=np.int16)
        self.__neighbors = np.zeros(num_edges, dtype=np.int16)
        self.__idx_edge = 0

    def add_edge(self, edge, neighbor):
        """
        Add edge to check node. 

            Parameters:
                edge (int): index of edge connected to check node
                neighbor (int): id of connected neighbor
            
            Returns:
                <none>
        """
        self.__edges[self.__idx_edge] = edge
        self.__neighbors[self.__idx_edge] = neighbor
        self.__idx_edge += 1

    def send_messages(self, messages):
        """
        Send messages as part of BP decoding. 

            Parameters:
                messages (ndarray): array of current factor graph messages

            Returns:
                <None>
        """
        incoming_messages = np.tanh(messages[self.__edges] / 2)
        in_product = np.product(incoming_messages)
        num_edges = len(self.__edges)

        if not np.isclose(in_product, 0):
            outMsgs = np.array([in_product / incoming_messages[i] 
                                for i in range(num_edges)])
        else:
            edges = np.arange(num_edges)
            outMsgs = np.array([np.product(
                                incoming_messages[np.setdiff1d(edges, i)]) 
                                for i in range(num_edges) ])
        
        messages[self.__edges] = 2 * np.arctanh(outMsgs)

    def check_consistency(self, estimate):
        """
        Check for parity consistency across neighboring nodes.

            Parameters:
                estimate (ndarray): hard decision bit estimates
            
            Returns:
                is_consistent (bool): flag of whether check is satisfied

        """
        is_consistent = (np.sum(estimate[self.__neighbors]) % 2 == 0)
        return is_consistent


class VariableNode():
    """
    Class for single variable node that implements LLR-domain message passing

        Attributes:
            outputLLR (double): output LLR for represented bit
        
        Methods:
            add_edge(edge): add edge to variable node
            set_observation(observation): set local (channel) observation
            send_message(messages): pass messages to neighboring check nodes
    """

    def __init__(self, num_edges):
        """
        Initialize VariableNode class.

            Parameters:
                num_edges (int): number of connected edges

            Returns:
                <none>
        """
        self.__edges = np.zeros(num_edges, dtype=np.int16)
        self.__idxEdge = 0
        self.output_llr = 0

    def add_edge(self, edge):
        """
        Add edge to variable node.

            Parameters:
                edge (int): index of edge connected to variable node
            
            Returns:
                <none>
        """
        self.__edges[self.__idxEdge] = edge
        self.__idxEdge += 1

    def set_observation(self, observation):
        """
        Set local (channel) observation.

            Parameters:
                observation (double): local (channel) observation
            
            Returns:
                <none>
        """
        self._observation = observation
    
    def send_messages(self, messages):
        """
        Pass messages to neighboring check nodes

            Parameters:
                messages (ndarray): array of current factor graph messages

            Returns:
                <none>
        """
        incoming_messages = messages[self.__edges]
        self.output_llr = np.sum(incoming_messages) + self._observation
        outgoing_messages = np.array([self.output_llr - incoming_messages[i]
                                      for i in range(len(incoming_messages))])
        messages[self.__edges] = np.clip(outgoing_messages, a_min=-15, a_max=15)


class Decoder():
    """
    Class for single-user, binary, LLR-domain LDPC decoder.
    
    Implements single-user, binary, LLR-domain LDPC decoder. This class is
    intended for simulating the performance of LDPC codes defined in the 
    alist format. 

        Attributes:
            N (int): length of the code
            K (int): dimension of the code
            M (int): number of checks
            messages (ndarray): array of factor graph messages
            variable_nodes (list): list of variable nodes within factor graph
            check_nodes (list): list of check nodes within factor graph

        Methods:
            reset(): reset all messages and observations within factor graph
            soft_decision_decoding(r, ec, nvar, max_iter): decode received sequence


    """

    def __init__(self, alist_file_name):
        """
        Initialize Decoder class. 

            Parameters:
                alist_file_name (str): name of file with alist code definition

            Returns:
                <none>
        """
        alist = open(alist_file_name, 'r')

        # Obtain N, M for LDPC code
        [N, M] = [int(x) for x in alist.readline().split()] 
        self.N, self.M = N, M
        self.K = N - M

        # Skip max left/right degrees
        _ = alist.readline()

        # Obtain left degrees for all variable nodes
        left_degrees = [int(x) for x in alist.readline().split()]
        assert len(left_degrees) == N
        
        # With left degrees, create message + variable node data structures
        num_edges = np.sum(left_degrees)
        self.messages = np.zeros(num_edges, dtype=np.float64)
        self.variable_nodes = [VariableNode(left_degrees[i]) for i in range(N)]

        # obtain right degrees for all check nodes
        right_degrees = [int(x) for x in alist.readline().split()]
        assert len(right_degrees) == M
        assert np.sum(right_degrees) == num_edges

        # Create check nodes with given degrees
        self.check_nodes = [CheckNode(right_degrees[i]) for i in range(M)]

        # Define connections between check and variable nodes
        idx_edge = 0
        for i in range(N):
            connections = [int(x) for x in alist.readline().split()]
            for j in connections:
                if j == 0: 
                    continue
                self.variable_nodes[i].add_edge(idx_edge)
                self.check_nodes[j-1].add_edge(idx_edge, i)
                idx_edge += 1

    def reset(self):
        """
        Reset all messages and local observations within factor graph.

            Parameters:
                <none>

            Returns:
                <none>
        """
        self.messages = np.zeros(self.messages.shape, dtype=np.float64)
        for varNode in self.variable_nodes:
            varNode.set_observation(0)

    def soft_decision_decoding(self, r, nvar, max_iter=100):
        """
        Perform soft decision decoding of received sequence. 

            Parameters:
                r (ndarray): unprocessed received vector
                nvar (double): AWGN noise variance
                max_iter (int): maximum BP iterations to perform

            Returns:
                cdwd_ht (ndarray): estimate of true codeword

            Constraints:
                len(r) == N

        """
        assert len(r) == self.N
        r = r.flatten()

        # Set channel observations
        for i in range(self.N): 
            self.variable_nodes[i].set_observation(-2*r[i] / nvar)

        # Initial variable to check node messages
        for i in range(self.N):
            self.variable_nodes[i].send_messages(self.messages)

        # Pass messages
        for idx_iter in range(max_iter):

            # Check to variable node messages
            for i in range(self.M):
                self.check_nodes[i].send_messages(self.messages)
            
            # Variable to check node messages
            for i in range(self.N):
                self.variable_nodes[i].send_messages(self.messages)
            
            # Make hard decisions on current codeword estimate 
            cdwd_ht = np.array([var_node.output_llr for var_node 
                                in self.variable_nodes])
            cdwd_ht = (-cdwd_ht > 0).astype(int)

            # Check for parity consistency of codeword estimate
            finished = True
            for i in range(self.M):
                if not self.check_nodes[i].check_consistency(cdwd_ht):
                    finished = False
                    break
            
            # End iterations early if codeword esimate is parity-consistent
            if finished: 
                break

        # return answer
        return cdwd_ht
