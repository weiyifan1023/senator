import random
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import pandas as pd
from tqdm import tqdm
from kg_rag.se import StructEntropy

from typing import List, Tuple, Dict


class KnowledgeGraph:
    def __init__(self, tok=None, model=None):
        self.graph = nx.MultiGraph()
        self.tokenizer = tok
        self.model = model

    def add_triple(self, head: str, relation: str, tail: str, w=None):
        """Add a triple to the knowledge graph."""
        self.graph.add_edge(head, tail, relation=relation, weight=w)

    # def get_relations(self, entity: str) -> List[str]:
    #     """Get all relations connected to an entity."""
    #     relations = set()
    #     for _, _, data in self.graph.out_edges(entity, data=True):
    #         relations.add(data['relation'])
    #     for _, _, data in self.graph.in_edges(entity, data=True):
    #         relations.add(f"inverse_{data['relation']}")
    #     return list(relations)

    # def get_neighbors(self, entity: str, relation: str) -> List[str]:
    #     """Get all neighboring entities connected by a specific relation."""
    #     neighbors = []
    #     if relation.startswith("inverse_"):
    #         relation = relation[8:]  # Remove "inverse_" prefix
    #         for src, _ in self.graph.in_edges(entity):
    #             if self.graph[src][entity].get('relation') == relation:
    #                 neighbors.append(src)
    #     else:
    #         for _, dst in self.graph.out_edges(entity):
    #             if self.graph[entity][dst].get('relation') == relation:
    #                 neighbors.append(dst)
    #     return neighbors

    def get_relations(self, entity: str) -> List[str]:
        """Get all relations connected to an entity."""
        relations = set()
        # 获取与实体连接的所有边（无方向的）
        for u, v, data in self.graph.edges(entity, data=True):
            if u == entity:
                relations.add(data['relation'])
            else:
                relations.add(f"inverse_{data['relation']}")
        return list(relations)

    def get_neighbors(self, entity: str, relation: str) -> List[str]:
        """Get all neighboring entities connected by a specific relation."""
        neighbors = []

        if relation.startswith("inverse_"):
            # Regularization: Avoid loops
            return []  # entity is a subject, so there are no inverse relations

            # Remove "inverse_" prefix
            relation = relation[8:]
            # Finds all edges associated with an entity
            for src, dst, data in self.graph.edges(entity, data=True):
                # If the edge's relation matches the specified relation and the direction is reversed
                if data.get('relation') == relation:
                    neighbors.append(src if dst == entity else dst)
        else:
            # Finds all edges associated with an entity
            for src, dst, data in self.graph.edges(entity, data=True):
                # If the edge's relation matches the specified relation and the direction is not reversed
                if data.get('relation') == relation:
                    neighbors.append(dst if src == entity else src)

        return neighbors

    def get_path(self, start: str, end: str, max_depth: int = 3) -> List[Tuple[str, str, str]]:
        """Find a path between two entities, returning a list of (head, relation, tail) triples."""
        path = nx.shortest_path(self.graph, start, end, weight=None)
        if len(path) > max_depth + 1:
            return []

        result = []
        for i in range(len(path) - 1):
            head, tail = path[i], path[i + 1]
            relation = self.graph[head][tail]['relation']
            result.append((head, relation, tail))
        return result

    def load_from_file(self, filename: str):
        """Load knowledge graph from a file (assuming tab-separated triples)."""
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                head, relation, tail = line.strip().split('\t')
                self.add_triple(head, relation, tail)

    def save_graph_to_df(self):
        """Save the knowledge graph to a pandas DataFrame with edge weights."""
        edge_data = []

        # 遍历图的边
        for source, target, data in self.graph.edges(data=True):
            relation = data['relation']
            weight = data['weight']

            # 将边的信息存储为字典
            edge_data.append({
                'source': source,
                'edge_type': relation,  # 对应 DataFrame 中的 'edge_type'
                'target': target,
                'weight': weight  # 新增的权重列
            })

        # 将边的信息转换为 DataFrame
        df_graph = pd.DataFrame(edge_data)
        df_graph.to_csv('/share/project/weiyifan/KG_RAG/data/qwen_kg.csv')
        print("Complete Save")

    def load_from_df(self, df):
        """Load knowledge graph from a pandas dataframe."""
        save_flag = False
        # 遍历 DataFrame，添加节点和边到图中
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            source = row['source']
            target = row['target']
            relation = row['edge_type']  # relation prompt

            if 'weight' in row:
                if pd.notna(row['weight']):  # 检查 'weight' 列的值是否不是 NaN
                    w = row['weight']
                else:
                    prompt = relation.format(source)
                    w = self.get_self_confidence(prompt, target)
            else:
                prompt = relation.format(source)
                w = self.get_self_confidence(prompt, target)
                save_flag = True
            # 添加边，不需要显式添加节点，因为 networkx 会自动添加
            self.graph.add_edge(source, target, relation=relation, weight=w)
        if save_flag:
            self.save_graph_to_df()


    def save_to_file(self, filename: str):
        """Save knowledge graph to a file (tab-separated triples)."""
        with open(filename, 'w', encoding='utf-8') as f:
            for head, tail, data in self.graph.edges(data=True):
                f.write(f"{head}\t{data['relation']}\t{tail}\t{data['weight']}\n")

    def get_self_confidence(self, query, answer):
        """
        This function is used to get the self-confidence P(a|q) of the LLM;
        Previous work use three model signals to represent the model’s confidence:
        1. Min-Prob, 2. Fst-Prob, 3. Prod-Prob.
        We select prod-prod that score the probability of the sum of all tokens.
        """

        # 将问题q和答案a拼接成模型的输入
        input_text = query + " " + answer

        # 对输入进行tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt")

        # 获取模型的输出logits
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 计算答案的概率P(a|q)
        # 模型输出的logits是预测每个token的概率
        logits = outputs.logits
        # 获取答案部分的token索引
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        # 计算答案部分的logits对应的log概率
        log_prob = 0
        prob = 0
        for i, token_id in enumerate(answer_ids):
            # 对应位置的logits的log概率
            index = len(inputs.input_ids[0]) - len(answer_ids)
            token_prob = torch.softmax(logits[0, index + i], dim=-1)
            prob += token_prob[token_id]
            token_log_prob = torch.log_softmax(logits[0, index + i], dim=-1)
            log_prob += token_log_prob[token_id]

        prob = prob / len(answer_ids)
        # print(f"P(a|q) = {prob.item()}, -log P(a|q) = {-log_prob.item()}")
        # return -log_prob.item()
        return prob.item()


class KGQuery:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def get_possible_actions(self, entity: str) -> List[str]:
        """Get all possible actions (relations) from a given entity."""
        return self.kg.get_relations(entity)

    def execute_action(self, entity: str, action: str) -> List[str]:
        """Execute an action (follow a relation) from a given entity."""
        return self.kg.get_neighbors(entity, action)

    def get_action4_entity(self, entity: str) -> List[Tuple[str, str]]:
        """
        Get all possible actions (relation-entity pairs) from a given entity.
        State: head ei, Action: (_, ri, tail ei), Next State: tail ei
        """
        search_width = 15  # 限制搜索宽度
        relations = self.kg.get_relations(entity)
        action_pairs = []
        for relation in relations:
            neighbors = self.kg.get_neighbors(entity, relation)
            for obj in neighbors:
                action_pairs.append((relation, obj))
        if len(action_pairs) > search_width:
            action_pairs = random.sample(action_pairs, search_width)
        return action_pairs

    def check_answer(self, entity: str, question: Dict) -> bool:
        """Check if the current entity is a valid answer to the question."""
        # This is a simplified implementation. In practice, you might need more sophisticated
        # answer checking logic, possibly involving the critical model.
        return entity in question.get('answer_entities', [])

    def get_context(self, entity: str) -> Dict[str, List[str]]:
        """Get the context (neighboring entities and relations) for a given entity."""
        context = {"relations": [], "entities": []}
        for relation in self.kg.get_relations(entity):
            context["relations"].append(relation)
            neighbors = self.kg.get_neighbors(entity, relation)
            context["entities"].extend(neighbors)
        return context

    def find_path(self, start: str, end: str, max_depth: int = 3) -> List[Dict[str, str]]:
        """Find a path between two entities, returning a list of steps."""
        path = self.kg.get_path(start, end, max_depth)
        return [{"from": h, "relation": r, "to": t} for h, r, t in path]


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0
        self.prior = 1  # 固定先验概率为1

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select_child(self):
        c_puct = 2.5
        return max(self.children,
                   key=lambda c: c.value / (c.visits + 1) + c_puct * c.prior * (self.visits ** 0.5) / (c.visits + 1))
        # 打印每个子节点的计算值
        # def compute_score(c):
        #     value_part = c.value / (c.visits + 1)
        #     exploration_part = c_puct * c.prior * (self.visits ** 0.5) / (c.visits + 1)
        #     return value_part + exploration_part
        #
        # return max(self.children, key=compute_score)

    def expand(self, actions, priors):
        for action, prior in zip(actions, priors):
            child = MCTSNode(state=None, parent=self, action=action)
            # child.prior = prior
            child.prior = 1  # 固定先验概率为1
            # adding tail entity to the current state
            child.state = action[1]
            self.children.append(child)

    def update(self, value):
        self.visits += 1
        self.value += value


class MCTS:
    def __init__(self, llm_model, kg_query: KGQuery = None, alg_se: StructEntropy = None,
                 num_simulations: int = 100, max_depth: int = 5):
        self.llm_model = llm_model
        self.kg_query = kg_query
        self.alg_se = alg_se
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.preferences = []
        self.root = MCTSNode(state=None)

    def search(self, root_state):
        self.root = MCTSNode(state=None)
        self.root.state = root_state
        root = self.root
        # root = MCTSNode(state=root_state)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            ### Selection
            while node.is_fully_expanded() and len(search_path) < self.max_depth:
                node = node.select_child()  # Greedily select next move.
                search_path.append(node)

            ### Expansion
            if len(search_path) < self.max_depth:
                # actions, priors = self.llm_model.get_action_probabilities(node.state)
                actions = self.kg_query.get_action4_entity(node.state)
                node.expand(actions, priors=[1] * len(actions))
                node = random.choice(node.children)
                search_path.append(node)

            ### Simulation: Evaluation
            # value = self.llm_model.evaluate(node.state)
            # reward value based on self-confidence
            # prompt = node.action[0].format(node.parent.state)
            # value = self.kg_query.kg.get_self_confidence(prompt, node.state)
            value = self.alg_se.calc_node_SE(node.state)
            # path_str = ""
            # for path in search_path:
            #     path_str += path.state + " ==> "
            # print("Reward Value: ", value, path_str)

            ### Backpropagation
            for node in reversed(search_path):
                node.update(value)

        # MCTS Search Over
        # best_path = self.collect_best_path()
        best_path = self.collect_paths_above_threshold()
        return root, best_path

    def get_action_probabilities(self, state, temperature=1):
        root = self.search(state)
        visits = [child.visits for child in root.children]
        actions = [child.action for child in root.children]

        if temperature == 0:
            best_index = visits.index(max(visits))
            probs = [0] * len(actions)
            probs[best_index] = 1
            return actions, probs

        visits = [v ** (1 / temperature) for v in visits]
        total = sum(visits)
        probs = [v / total for v in visits]

        return actions, probs

    def collect_best_path(self):
        def traverse(node):
            # 如果当前节点没有子节点，说明已经到达叶子节点，返回路径
            if not node.children:
                return [node]

            # 选择具有最高平均奖励值的子节点
            best_child = max(node.children, key=lambda c: c.value if c.value > 0 else 0)

            # 递归遍历最佳子节点，并将当前节点添加到路径中
            return [node] + traverse(best_child)

        # 从根节点开始遍历，获取最佳路径
        best_path = traverse(self.root)
        return best_path

    def collect_paths_above_threshold(self, m=2500):
        def traverse(node, current_path, current_reward):
            # 如果当前节点没有子节点，说明已经到达叶子节点
            if not node.children:
                # 如果当前路径的总奖励值大于阈值m，返回当前路径
                if current_reward > m:
                    return [current_path]
                return []

            # 递归遍历所有子节点，收集所有满足条件的路径
            paths = []
            for child in node.children:
                # 计算当前子节点的路径奖励值
                new_reward = current_reward + (child.value if child.value > 0 else 0)
                # 递归查找路径
                paths.extend(traverse(child, current_path + [child], new_reward))

            return paths

        # 从根节点开始遍历，获取所有符合条件的路径
        all_paths = traverse(self.root, [self.root], 0)
        return all_paths


    def collect_preferences(self, root):
        def traverse(node):
            if node.children:
                best_child = max(node.children, key=lambda c: c.value / c.visits if c.visits > 0 else 0)
                worst_child = min(node.children, key=lambda c: c.value / c.visits if c.visits > 0 else float('inf'))
                self.preferences.append((node.state, best_child.action, 1))
                self.preferences.append((node.state, worst_child.action, 0))
                for child in node.children:
                    traverse(child)

        traverse(root)

    def hdpo_loss(self, logits, values):
        preferred_logits = logits[:, 0]
        dispreferred_logits = logits[:, 1]
        preferred_values = values[:, 0]
        dispreferred_values = values[:, 1]

        policy_loss = -torch.log(torch.sigmoid(preferred_logits - dispreferred_logits))
        value_loss = torch.max(torch.zeros_like(preferred_values), 0.1 - (preferred_values - dispreferred_values))
        reg_loss = sum(p.pow(2.0).sum() for p in self.llm_model.parameters())

        return policy_loss.mean() + value_loss.mean() + 0.01 * reg_loss

    def fine_tune_llm(self):
        optimizer = optim.Adam(self.llm_model.parameters(), lr=1e-5)
        self.llm_model.train()

        for preference in self.preferences:
            state, action, value = preference

            input_text = f"State: {state}, Action: {action}"
            inputs = self.llm_model.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

            policy_output, value_output = self.llm_model(inputs.input_ids, inputs.attention_mask)

            loss = self.hdpo_loss(policy_output, value_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.preferences = []
