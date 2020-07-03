import numpy as np
import copy

def dict_product(l1, l2):
    if (l1 == []):
        return l2
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
                'conprob' : [],
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
                    self.net[node]['conprob'].append(
                        {
                            node: self.net[node]['domain'][i],
                            'prob' : self.net[node]['prob'][i]
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
                    for d in tmp_dict:
                        d.update(
                            {
                                node : val
                            }
                        )
                        self.net[node]['conprob'].append(d)
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
            prob = self.net[node]['conprob']
            self.factors.append((x, prob)) 
        

    def elim_vars(self, query_variables, evidence_variables):    
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
        
        elim_vars = []
        for node in self.net:
            if node not in factors:
                elim_vars.append(node)
        
        
        return elim_vars

    def sum_product(self, elim_vars):
        """
        Eliminate the non-query and non-evidence from factor set
        
        args:
            elim_factors
        
        return:
            sum_product
        """ 
        phi_star = self.factors
        # pr = self.product_factors(self.factors[0], self.factors[1])
        # sum_by_i = self.sum_by_var('I', pr)
        # print(sum_by_i)
        print(elim_vars)
        for var in elim_vars:
            all_phi = copy.deepcopy(phi_star)
            print("=====================BEGIN=============================")
            print(var)
            print("=============")
            factor_star = None
            for factor in all_phi:
                vars, probs = factor
                if var in vars:
                    phi_star.remove(factor)
                    factor_star = self.product_factors(factor_star, factor)
            factor_star = self.sum_by_var(var, factor_star)
            phi_star.append(factor_star)
            print(phi_star)
            print("========================END=========================")

    
    def product_factors(self, f1, f2):
        """
        product two factor of network

        args:
            f1: first factor
            f2: second factor
        
        return:
            pr: production of both
        """
        if f1 is None:
            return f2
        vars1, cases1 = f1
        vars2, cases2 = f2
        pr_vars = vars1
        for var in vars2:
            if var not in pr_vars:
                pr_vars.append(var)

        pr_prob = []
        for case1 in cases1:
            for case2 in cases2:
                can_product = True
                for key1, val1 in case1.items():
                    if key1 in vars2 and key1 != 'prob':
                        for key2, val2 in case2.items():
                            if key1 == key2:
                                if val1 != val2:
                                    can_product = False
                                    break
                product = 0
                if (can_product):
                    product = case1['prob'] * case2['prob']
                    new_case = {**case1, **case2}
                    new_case['prob'] = product
                    pr_prob.append(new_case)

        return pr_vars, pr_prob


    def sum_by_var(self, var, factor):
        """
        eliminate the var from the factor

        args:
            var: the var that  should be eliminated after this
            factor: the factor contains that var

        return:
            sum_by_var: neew factor without var
        """
        vars, cases = factor
        res = []
        sum_same_val = { }
        for org_case in cases:
            sum_same_val = org_case
            cases.remove(org_case)
            del sum_same_val[var]
            for case in cases:
                can_sum = True
                for key, val in case.items():
                    if key != 'prob' and key != var:
                        if org_case[key] != case[key]:
                            can_sum = False
                            break
                if can_sum:
                    sum_prob = sum_same_val['prob'] + case['prob']
                    sum_same_val = {**case, **sum_same_val}
                    sum_same_val['prob'] = sum_prob
                    del sum_same_val[var]
                    cases.remove(case)
            tmp = copy.deepcopy(sum_same_val)
            res.append(tmp)
        vars.remove(var)
        return (vars , res)


    def exact_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        query_variables, evidence_variables = self.__extract_query(f.readline())
        # YOUR CODE HERE
        elim_vars = self.elim_vars(query_variables, evidence_variables)
        self.sum_product(elim_vars)

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
