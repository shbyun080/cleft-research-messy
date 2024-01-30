import mysql.connector


# Replaces data points on image_file's row with new data
# NEED TESTING
def insert_data(db, image_file, data):

    cur = db.cursor()

    cur.execute("""SELECT id from cleftpoints WHERE imagefile='{}'""".format(image_file))
    result = cur.fetchall()

    f_data = [None if i==0 else i for i in data]
    str_data = ["NULL" if i == 0 else str(i) for i in data]

    if len(result) == 0:  # if such row doesn't exist
        insert_state = "INSERT INTO cleftpoints VALUES (" + ", ".join(["%s"]*(2+len(str_data))) + ")"
        f_data.insert(0, image_file)
        f_data.insert(0, None)
        cur.execute(insert_state, f_data)
    else:
        cur.execute("""REPLACE INTO cleftpoints VALUES ({}, '{}', {})""".format(result[0][0], image_file, ', '.join(str_data)))

    db.commit()
    cur.close()


# Fetches stored data points for an image
def fetch_data(db, image_file):
    cur = db.cursor()
    cur.execute("""SELECT * from cleftpoints WHERE imagefile='{}'""".format(image_file))
    result = cur.fetchall()
    cur.close()
    if len(result) == 0:
        return None
    return result[0][2:]


# Returns a queryable connection to the AWS RDS database
def get_db():
    mydb = mysql.connector.connect(
        host="cleftmarker-db.ccyyssjtizqk.us-west-1.rds.amazonaws.com",
        port="3306",
        user="root",
        password="01220828",
        database="cleftmarker_1"
    )
    return mydb


if __name__ == "__main__":
    print(fetch_data(get_db(), "abc"))