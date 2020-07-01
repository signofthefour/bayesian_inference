import numpy as np
import copy

def dict_product(l1, l2):
    res = []
    for x in l1:
        for y in l2:
            res.append(dict(x, **y))
    return res


class BayesianNetwork:
    def __init__(self, filename):
        f = open(filename, 'r') 
        N = int(f.readline())
        lines = f.readlines()
        self.net = {}
        for line in lines:
            node, parents, domain, shape, probabilities = self.__extract_model(line)
            # YOUR CODE HERE
            for p in parents:
                self.net[p]['children'].append(node)
            self.net[node]  = {
                'parents' : parents,
                'domain' : domain,
                'shape' : shape,
                'prob' : probabilities,
                'conprob' : {},
                'children' : []
            }
        f.close()
        """
        Create a condition probability dictionary which can provide an easy
        way to access the prob by query_ and evidence_var:
        conprob = {
            node->domain[i] : [
                    {
                        parent[j1] : parent[j1]->domain[k1]
                        parent[j2] : parent[j2]->domain[k2]
                        ...
                        condition_prob : prob
                    },
                ...
            ]
            ...
        }
        """
        for node in self.net:
            if len(self.net[node]['parents']) == 0:
                for i in range(self.net[node]['shape']):
                    self.net[node]['conprob'].update(
                        {
                            self.net[node]['domain'][i] : [
                                    {
                                        'prob' : self.net[node]['prob'][i]
                                    }
                                ]
                        }
                    )
            else:
                cases = []
                for p in self.net[node]['parents']:
                    p_condition_cases = []
                    for val in self.net[p]['domain']:
                        p_condition_cases.append (
                            {p : val}
                        )
                    cases.append(p_condition_cases)

                if len(cases) > 1:
                    for i in range(1, len(cases)):
                        cases[0] = dict_product(cases[0], cases[i])
                
                prob = self.net[node]['prob'].flatten()
                count = 0
                domain_length = self.net[node]['shape'][-1]

                for val in self.net[node]['domain']:
                    idx = self.net[node]['domain'].index(val)
                    for i in range(len(cases[0])):
                        cases[0][i].update(
                            {
                                'prob' : prob[idx + domain_length * i]
                            }
                        )
                    tmp_dict = copy.deepcopy(cases[0])
                    self.net[node]['conprob'].update(
                        {
                            val : tmp_dict
                        }
                    )
        self.make_factor()
                
    """
    Already in topological order
    """ 

    def make_factor(self):
        self.factors = []
        for node in self.net:
            x = []
            x.append(node)
            for p in self.net[node]['parents']:
                x.append(p)
            self.factors.append(x)
        

    def elim_factor(self, query_variables, evidence_variables):    
        """
        Get the factor that need eliminated

        args:
            query_variables
            evidence_variables
        
        return:
            elim_factors
        """
        factors = []
        for key, val in query_variables.items():
            factors.append(key)
        for key, val in evidence_variables.items():
            factors.append(key)
        
        elim_factors = []
        for node in self.net:
            if node not in factors:
                elim_factors.append(node)
        
        return elim_factors

    def sum_product(self, elim_factors):
        """
        Eliminate the non-query and non-evidence from factor set
        
        args:
            elim_factors
        
        return:
            sum_product
        """
        
        print(elim_factors)



    def exact_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        query_variables, evidence_variables = self.__extract_query(f.readline())
        # YOUR CODE HERE
        elim_factors = self.elim_factor(query_variables, evidence_variables)
        self.sum_product(elim_factors)

        f.close()
        return result

    def approx_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        # YOUR CODE HERE


        f.close()
        return result

    def __extract_model(self, line):
        parts = line.split(';')
        node = parts[0]
        if parts[1] == '':
            parents = []
        else:
            parents = parts[1].split(',')
        domain = parts[2].split(',')
        shape = eval(parts[3])
        probabilities = np.array(eval(parts[4])).reshape(shape)
        return node, parents, domain, shape, probabilities

    def __extract_query(self, line):
        parts = line.split(';')

        # extract query variables
        query_variables = {}
        for item in parts[0].split(','):
            if item is None or item == '':
                continue
            lst = item.split('=')
            query_variables[lst[0]] = lst[1]

        # extract evidence variables
        evidence_variables = {}
        for item in parts[1].split(','):
            if item is None or item == '':
                continue
            lst = item.split('=')
            evidence_variables[lst[0]] = lst[1]
        return query_variables, evidence_variables
