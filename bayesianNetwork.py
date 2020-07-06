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

    def sum_product(self, elim_vars, factors):
        """
        Eliminate the non-query and non-evidence from factor set
        
        args:
            elim_factors
        
        return:
            sum_product
        """ 
        phi_star = factors
        # pr = self.product_factors(self.factors[0], self.factors[1])
        # sum_by_i = self.sum_by_var('I', pr)
        # print(sum_by_i)
        for var in elim_vars:
            all_phi = copy.deepcopy(phi_star)
            factor_star = None
            for factor in all_phi:
                vars, probs = factor
                if var in vars:
                    phi_star.remove(factor)
                    factor_star = self.product_factors(factor_star, factor)
            factor_star = self.sum_by_var(var, factor_star)
            phi_star.append(factor_star)

        phi = None
        for factor in phi_star:
            phi = self.product_factors(phi, factor)
 
        return phi

    
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
        pr_vars = copy.deepcopy(vars1)
        for var in vars2:
            if var not in pr_vars:
                pr_vars.append(var)

        has_product = False
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
                    has_product = True
                    product = case1['prob'] * case2['prob']
                    new_case = {**case1, **case2}
                    new_case['prob'] = product
                    pr_prob.append(new_case)

        if has_product == False:
            pr_vars = vars1
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
        while (len(cases) > 0):
            org_case = cases[0]
            cases.remove(org_case)
            case_same_val = []
            for case in cases:
                can_sum = True
                for key, val in case.items():
                    if key != 'prob' and key != var:
                        if val != org_case[key]:
                            can_sum = False
                            break
                if can_sum:
                    org_case['prob'] += case['prob']
                    case_same_val.append(case)
            for case in case_same_val:
                cases.remove(case)
            tmp = copy.deepcopy(org_case)
            del tmp[var]
            res.append(tmp)

        vars.remove(var)
        return (vars , res)

    def elim_by_evidence(self, evidence_variables, factors):
        knew_evidence_keys = [key for key in evidence_variables.keys()]
        res = []
        factors = copy.deepcopy(factors)
        for factor in factors:
            vars, cases = factor
            new_cases = copy.deepcopy(cases)
            for case in cases:
                factor_keys = [key for key in case.keys()]
                can_del = False
                for key in factor_keys:
                    if key in knew_evidence_keys:
                        if evidence_variables[key] == case[key]:
                            continue
                        else:
                            can_del = True
                            break
                if can_del:
                    new_cases.remove(case)
            res.append((vars, new_cases))
        return res

    def exact_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        query_variables, evidence_variables = self.__extract_query(f.readline())
        # YOUR CODE HERE
        elim_vars = self.elim_vars(query_variables, evidence_variables)
        if evidence_variables == { }:
            factors = copy.deepcopy(self.factors)
            phi_star = self.sum_product(elim_vars, factors)
            vars, cases = phi_star
            all_keys = [key for key in query_variables.keys()]
            for case in cases:
                is_result = True
                for key in all_keys:
                    if case[key] != query_variables[key]:
                        is_result = False
                        break
                if is_result:
                    result =case['prob']
                    break 
        else:
            factors = copy.deepcopy(self.factors)
            new_factors = self.elim_by_evidence(evidence_variables, factors)
            phi_star = self.sum_product(elim_vars, new_factors)
            all_evidence_keys = [key for key in evidence_variables.keys()]
            all_query_keys = [key for key in query_variables.keys()]
            vars, factors = phi_star
            alpha = 0
            for factor in factors:
                is_result = True
                for key in all_evidence_keys:
                    if factor[key] != evidence_variables[key]:
                        # We donnot have to handle the case that query and evidence have same key
                        # cause its donot have knowledge value
                        is_result = False
                        break
                if is_result:
                    alpha += factor['prob']
                    for key in all_query_keys:
                        if factor[key] != query_variables[key]:
                            is_result = False
                            break
                if is_result:
                    result = factor['prob']
            
            result = result / alpha

        f.close()
        return result


    def sampling_given_distribution(self, evidence):
        """
        sampling a new sample from an initial state

        args: 
            evidence : initial for cur_sample
        
        return:
            cur_sample: list of sample that is sampled by forward
        """
        cur_sample = evidence
        all_nodes = [node for node  in  self.net]
        factors = copy.deepcopy(self.factors)
        for i in range(len(factors)):
            node = factors[i][0][0]
            cur_table = self.elim_by_evidence(cur_sample, [factors[i]])
            all_nodes, cases = cur_table[0]
            cur_distri = []
            domain = []
            for case in cases:
                cur_distri.append(case['prob'])
                if case[ node ] not in domain:
                    domain.append(case[ node ])
            sigma = np.std(cur_distri)
            mu = np.mean(cur_distri)
            cdf = np.cumsum(cur_distri)
            random_val = np.random.normal(mu, sigma)
            j = 0
            for j in range(len(cdf)):
                if random_val < cdf[j]:
                    break
            
            cur_sample.update( {
                node : domain[j]
            })
                
        # distribution = np.zeros(100)
        # mu, sigma = 0, 0.1
        # distribution = np.random.normal(mu, sigma, 5000)
        # distribution = np.reshape(distribution, (1000, 5))
        # print(distribution)
        # import matplotlib.pyplot as plt
        # count, bins, ignored = plt.hist(distribution                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    , 30, density=True)
        # plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
        # plt.show()
        
        # for node in self.net:
        #     for val in self.net[node]['domain']:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

        # print(cur_sample)
        return cur_sample
    


    def gibbs_sampling(self, factors, X_s, P0, T):
        """
        
        Use gibbs sampling algorithm to generate T new samples from first init P0

        args:
            factors: all factors  with their cumulative distribution function
            X_s: sampling variables
            P0: first state of chain
            T: number of iterations

        return: list of samples
        """
        init_factors = copy.deepcopy(factors)
        res = []
        P_cur = P0
        # P_cur = {
        #     'I' : 'Cao',
        #     'D' : 'Kho',
        #     'S' : 'Cao',
        #     'L' : 'Manh',
        #     'G' : 'A'
        # }
        P_new = []
        for i in range(T):
            P_new = copy.deepcopy(P_cur)
            for X in X_s:
                P_temp = copy.deepcopy(P_new)
                del P_temp[X]
                elimed_factors = self.elim_by_evidence(P_temp, init_factors)
                table = None
                for factor in elimed_factors:
                    table = self.product_factors(table, factor)
                distribution = []
                domain = []
                keys, cases = table
                for case in cases:
                    distribution.append(case['prob'])
                    if case[X] not in domain:
                        domain.append(case[X])

                dis_sum = np.sum(distribution)
                distribution = [val / dis_sum for val in distribution]
                sigma = np.std(distribution)
                mu = np.mean(distribution)
                cdf = np.cumsum(distribution)
                random_val = np.random.normal(mu, sigma)
                j = 0
                for j in range(len(domain)):
                    if random_val < cdf[j]:
                        break
                P_new[X] = domain[j]
                del P_temp
            P_temp = copy.deepcopy(P_new)
            res.append(P_temp)
        return  res


    def approx_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        query_variables, evidence_variables = self.__extract_query(f.readline())
        # YOUR CODE HERE
        nodes = [node for  node in self.net]
        evidence_key = [key for key in evidence_variables.keys()]
        sampling_var = []
        for node in nodes:
            if node not in evidence_key:
                sampling_var.append(node)
        P0 = self.sampling_given_distribution(evidence_variables)
        T = 3000
        res = self.gibbs_sampling(self.factors, sampling_var, P0, T)
        count = 0
        query_key = [key for key in query_variables.keys()]
        for case in res:
            is_query_case = True
            for key, val in case.items():
                if key in query_key:
                    if val != query_variables[key]:
                        is_query_case = False
                        break
            if is_query_case:
                count+=1

        result = count / T
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
