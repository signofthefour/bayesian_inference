import numpy as np

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
                for i in range(self.net[node]['shape'][-1]):
                    self.net[node]['conprob'].update(
                        {
                            self.net[node]['domain'][i] : []
                        }
                    )
                    
            print(self.net[node]['conprob'])

    """
    Already in topological order
    """ 

    def exact_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        query_variables, evidence_variables = self.__extract_query(f.readline())
        # YOUR CODE HERE
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
