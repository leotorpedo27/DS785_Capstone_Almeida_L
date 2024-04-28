SELECT *
FROM fuzzy_match.predictions_v2 f
LEFT JOIN fuzzy_match.mdm_customer_masters m ON f.MDM_CUST_KEY = m.MDM_CUST_KEY
ORDER BY f.lead_gen_prob