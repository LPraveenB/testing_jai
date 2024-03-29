LOCATION_GROUP_FOLDER_PREFIX = "LOCATION_GROUP="
DATA_SPLIT_FOLDER_PREFIX = "data_split="
DATA_SPLIT_TEST_DATA_FOLDER = DATA_SPLIT_FOLDER_PREFIX + "test"
LOAD_DATE_FOLDER_PREFIX = "LOAD_DATE="
PARQUET_FILE_EXTENSION = ".parquet"
BINARY_PRED = "binary_pred"
ORIG_PRED = "orig_pred"
ORIG_PRED_TEMP = "orig_pred_temp"
ORIG_PRED_LOW = "orig_pred_low"
ORIG_PRED_HIGH = "orig_pred_high"
HL_ORIG_PRED = "hl_orig_pred"
OOS = "OOS"
STEP_1 = "step_1"
STEP_2 = "step_2"
STEP_3 = "step_3"
STEP_4 = "step_4"
FINAL_PREDICTION = "final_prediction"

SKU = 'SKU'
LOCATION = 'LOCATION'
DATE = 'DATE'
LOCATION_GROUP = 'LOCATION_GROUP'
TARGET_CLASSIFICATION_BINARY = "BINARY_TARGET_CLASSIFICATION"
TARGET_CLASSIFICATION_BINARY_HL = "BINARY_HL_TARGET_CLASSIFICATION"
TARGET_REGRESSION = "TARGET_REGRESSION"
TARGET_REGRESSION_H = "H_REV_TARGET_RATIO"
IP_QTY_EOP_SOH = "IP_QTY_EOP_SOH"
REV_IP_QTY_EOP_SOH = "REV_IP_QTY_EOP_SOH"
CURDAY_IP_QTY_EOP_SOH = "CURDAY_IP_QTY_EOP_SOH"
TOTAL_RETAIL = "TOTAL_RETAIL"
Y = "Y"
N = "N"

REQ_COLUMNS_OOS_CLASSIFIER = [SKU, LOCATION, DATE, BINARY_PRED, TARGET_CLASSIFICATION_BINARY, IP_QTY_EOP_SOH,
                              CURDAY_IP_QTY_EOP_SOH, REV_IP_QTY_EOP_SOH, TOTAL_RETAIL]
REQ_COLUMNS_HL_CLASSIFIER = [SKU, LOCATION, DATE, HL_ORIG_PRED, TARGET_CLASSIFICATION_BINARY_HL]
REQ_COLUMNS_REGRESSION_LB = [SKU, LOCATION, DATE, ORIG_PRED, ORIG_PRED_LOW, ORIG_PRED_HIGH, TARGET_REGRESSION]
REQ_COLUMNS_REGRESSION_HB = [SKU, LOCATION, DATE, ORIG_PRED, ORIG_PRED_LOW, ORIG_PRED_HIGH, TARGET_REGRESSION_H]
REQ_COLUMNS_FINAL_PREDICTION = [SKU, LOCATION, DATE, IP_QTY_EOP_SOH, CURDAY_IP_QTY_EOP_SOH, REV_IP_QTY_EOP_SOH,
                                TOTAL_RETAIL, OOS, BINARY_PRED, HL_ORIG_PRED, ORIG_PRED,
                                ORIG_PRED_LOW, ORIG_PRED_HIGH, TARGET_CLASSIFICATION_BINARY]