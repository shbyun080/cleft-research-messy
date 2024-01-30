import mysql.connector
import backend


def setup_db():
    mydb = backend.get_db()
    cur = mydb.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS cleftpoints(
        id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        imagefile VARCHAR(100) UNIQUE NOT NULL,
        upperleft_x FLOAT,
        upperleft_y FLOAT,
        lowerright_x FLOAT,
        lowerright_y FLOAT,
        cala_x FLOAT,
        cala_y FLOAT,
        sn_x FLOAT,
        sn_y FLOAT,
        ccphs_x FLOAT,
        ccphs_y FLOAT,
        ncch_x FLOAT,
        ncch_y FLOAT,
        ncala_x FLOAT,
        ncala_y FLOAT,
        ncchps_x FLOAT,
        ncchps_y FLOAT,
        cc_x FLOAT,
        cc_y FLOAT,
        csbal_x FLOAT,
        csbal_y FLOAT,
        ncsbal_x FLOAT,
        ncsbal_y FLOAT,
        cch_x FLOAT,
        cch_y FLOAT,
        ncc_x FLOAT,
        ncc_y FLOAT,
        prn_x FLOAT,
        prn_y FLOAT,
        ls_x FLOAT,
        ls_y FLOAT,
        mchpi_x FLOAT,
        mchpi_y FLOAT,
        crl_x FLOAT,
        crl_y FLOAT,
        sto_x FLOAT,
        sto_y FLOAT,
        cnt1_x FLOAT,
        cnt1_y FLOAT,
        cnt2_x FLOAT,
        cnt2_y FLOAT,
        lchpi_x FLOAT,
        lchpi_y FLOAT,
        mrl_x FLOAT,
        mrl_y FLOAT,
        ccphi_x FLOAT,
        ccphi_y FLOAT
    )""")


if __name__ == "__main__":
    setup_db()
