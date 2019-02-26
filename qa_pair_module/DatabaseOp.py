def get_questions(ids, tablename, mycursor):
    """
    根据倒排表检索候选答案
    :param ids: Set()
    :param tablename: str
    :param mycursor:
    :return: List(str)
    """
    sql = "SELECT content from %s" % tablename

    questions = []
    for i in ids:
        mycursor.execute(sql+" where que_id='%s'" % i)
        data = mycursor.fetchone()
        questions.append(data[0].strip())

    return questions


def get_answers(ids, mycursor):
    """
    根据问题id检索出对应答案
    :param ids: List(int)
    :return: List(List(tuple(int,str)))
    """
    answers = []
    for i in ids:
        sql = "select ans_id, content from answers where que_id=%d" % int(i)
        mycursor.execute(sql)
        temp = []
        for (j, ans) in mycursor.fetchall():
            temp.append((j, ans.strip()))
        answers.append(temp)

    return answers
