import mysql.connector as mc


def build_revise_ranking_dict(tablename):
    """
    建立倒排表
    :return: List(tuple(str,int))
    """
    import jieba
    import jieba.analyse as analyse
    jieba.load_userdict(userdict)
    analyse.set_stop_words(stopwords_path)

    retrieval_index = []
    sql = "SELECT que_id, content from %s" % tablename
    mycursor.execute(sql)
    data = mycursor.fetchall()

    for i, val in data:
        words = list(analyse.extract_tags(val.strip(), topK=20))

        for w in words:
            retrieval_index.append((w, int(i)))

    return retrieval_index


def build_table(tablename):
    sql = "CREATE TABLE %s(" \
          "id INT UNSIGNED AUTO_INCREMENT," \
          "word VARCHAR(100) NOT NULL," \
          "que_id INT UNSIGNED NOT NULL," \
          "PRIMARY KEY(id)," \
          "INDEX word(word))" % tablename
    mycursor.execute(sql)


def import_mysql(lst, tablename):
    """
    将倒排表导入数据库中
    :param :List(tuple)
    :return:
    """
    sql = "INSERT INTO %s(word, que_id)" % tablename + " VALUES (%s, %s)"
    values = []
    total = 1
    for key, val in lst:
        values.append((key, int(val)))
        total += 1
    mycursor.executemany(sql, values)
    mydb.commit()
    print("Insert total %d values" % total)


def start(name):
    build_table(name + back)
    result = build_revise_ranking_dict(name)
    import_mysql(result, name + back)


if __name__ == '__main__':
    userdict = 'vocab.txt'
    stopwords_path = 'stop_words.utf8'
    # surgical = "surgical"
    # internalMedical = "internalMedicine"
    # gynecology = "gynecology"
    # comprehensive1 = "comprehensive1"
    # comprehensive2 = "comprehensive2"
    back = "_index"

    mydb = mc.connect(host='localhost', user='root', password='123456789', database='medical',
                      auth_plugin='mysql_native_password')
    mycursor = mydb.cursor()

    start('questions')
