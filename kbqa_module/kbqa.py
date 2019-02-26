#!/usr/bin/env python3
# coding: utf-8

from kbqa_module.entity_extractor import EntityExtractor
from kbqa_module.search_answer import AnswerSearching


class KBQA:
    def __init__(self):
        self.extractor = EntityExtractor()
        self.searcher = AnswerSearching()

    def qa_main(self, input_str):
        answer = "对不起，您的问题我不知道，我今后会努力改进的。"
        entities = self.extractor.extractor(input_str)
        if not entities or entities['intentions'] == 'QA_matching':
            return "qa matching"
        sqls = self.searcher.question_parser(entities)
        final_answer = self.searcher.searching(sqls)
        if not final_answer:
            return answer
        else:
            return '\n'.join(final_answer)

    def entity_intent_reg(self, input_str):
        """
        抽取实体并识别意图
        :param input_str:str 问题
        :return:
        """
        entities = self.extractor.extractor(input_str)
        return entities

    def search(self, entity_intent):
        """
        检索答案
        :param entity_intent: Dict,实体和意图
        :return: str
        """
        answer = "对不起，您的问题我不知道，我今后会努力改进的。"
        sqls = self.searcher.question_parser(entity_intent)
        final_answer = self.searcher.searching(sqls)
        if not final_answer:
            return answer
        else:
            return '\n'.join(final_answer)
