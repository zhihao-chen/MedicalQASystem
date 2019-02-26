def bool_retrieval(word, tablename, mycursor):
    sql = "select que_id from %s where word='%s'" % (tablename, word)
    mycursor.execute(sql)

    res = set()
    for i in mycursor.fetchall():
        #yield int(i[0])
        res.add(int(i[0]))
    return res


def get_candidate(ids, tablename, mycursor):
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
        mycursor.excute(sql+"where que_id='%s' " % i)
        data = mycursor.fetchone()
        questions.append(data[0].strip())

    return questions
