# Legacy v1 column map

| v1 header (2014-2017) | v1 header (2018-2021) | Meaning |
|-----------------------|-----------------------|---------|
| `CUST_NO` | `CustomerId` | Customer identifier (string, zero-padded) |
| `NAME_1` | `LastName` | Last name (latin-1 accents) |
| `NAME_2` | `FirstName` | First name |
| `DT_CREAT` | `CreatedOn` | Creation date, DD/MM/YYYY |
| `AMT` | `Balance` | Balance in cents (integer; `-999` = unknown) |
| `FLAG_A` | `Active` | `O`/`N` in 2014-2017 files, `Y`/`N` later |

Files never mix eras: the first header cell tells you which map applies.
