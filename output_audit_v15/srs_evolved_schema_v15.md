# Master Payment SRS v15

SDNF is demonstrated in the Payment domain as a representative high-stakes semantic integration setting. Schema descriptors provide intended contracts; payloads provide empirical evidence. The Master Payment SRS evolves through normal-form-governed decisions and produces explainable payload compliance decisions before payment initiation.

## Overview
- Schema count: 8
- Schema attributes ingested: 61
- Canonical node count: 41
- Payload compliance records: 64
- Candidate schema deltas: 0

## Canonical Concepts
### account_identifier
- Semantic family: identifier:plaid_account
- Members: Plaid.account_id
- Rails: open_banking
### account_type
- Semantic family: payment_account:account_type
- Members: Plaid.account_type
- Rails: open_banking
### bank_account_number
- Semantic family: payment_account:account_number
- Members: Plaid.account_number
- Rails: open_banking
### card_acceptor_identifier
- Semantic family: identifier:card_acceptor
- Members: Mastercard.card_acceptor_id
- Rails: card_network_iso8583
### card_brand
- Semantic family: payment_card:brand
- Members: Stripe.card_brand
- Rails: psp_gateway
### card_expiration_date
- Semantic family: payment_card:expiration_date
- Members: INAmex.ExpirationDate, PPVisa.exp
- Rails: card_payment
### card_issuer
- Semantic family: payment_card:issuer
- Members: PPVisa.issuer
- Rails: card_payment
### card_last4
- Semantic family: payment_card:last4
- Members: Stripe.card_last4
- Rails: psp_gateway
### card_primary_account_number
- Semantic family: payment_card:pan
- Members: INAmex.PrimaryAccountNumber, Mastercard.primary_account_number, PPVisa.pan
- Rails: card_network_iso8583, card_payment
### card_verification_value
- Semantic family: payment_card:verification_value
- Members: INAmex.CardVerificationValue, PPVisa.cvv
- Rails: card_payment
### cardholder_name
- Semantic family: party:cardholder_name
- Members: INAmex.CardHolderName, PPVisa.cardholder
- Rails: card_payment
### created_at
- Semantic family: temporal:created_at
- Members: Razorpay.created_at, Stripe.created
- Rails: psp_gateway
### creditor_account
- Semantic family: payment_account:creditor_account
- Members: ISO20022.cdtr_acct
- Rails: account_to_account_credit_transfer
### creditor_name
- Semantic family: party:creditor_name
- Members: ISO20022.cdtr_nm
- Rails: account_to_account_credit_transfer
### current_balance
- Semantic family: payment_account:current_balance
- Members: Plaid.balance_current
- Rails: open_banking
### customer_contact
- Semantic family: party:customer_contact
- Members: Razorpay.contact
- Rails: psp_gateway
### customer_email
- Semantic family: party:customer_email
- Members: Razorpay.email
- Rails: psp_gateway
### customer_identifier
- Semantic family: identifier:customer
- Members: Stripe.customer_id
- Rails: psp_gateway
### debtor_account
- Semantic family: payment_account:debtor_account
- Members: ISO20022.dbtr_acct
- Rails: account_to_account_credit_transfer
### debtor_name
- Semantic family: party:debtor_name
- Members: ISO20022.dbtr_nm
- Rails: account_to_account_credit_transfer
### end_to_end_identifier
- Semantic family: identifier:end_to_end_payment
- Members: ISO20022.end_to_end_id
- Rails: account_to_account_credit_transfer
### merchant_category_code
- Semantic family: merchant:category_code
- Members: Mastercard.merchant_category_code
- Rails: card_network_iso8583
### message_identifier
- Semantic family: identifier:message
- Members: ISO20022.msg_id
- Rails: account_to_account_credit_transfer
### order_identifier
- Semantic family: identifier:order
- Members: Razorpay.razorpay_order_id
- Rails: psp_gateway
### payee_virtual_payment_address
- Semantic family: upi:payee_vpa
- Members: UPI.payee_vpa
- Rails: upi
### payer_account
- Semantic family: payment_account:payer_account
- Members: UPI.payer_account
- Rails: upi
### payment_amount
- Semantic family: payment:amount
- Members: ISO20022.instd_amt, Mastercard.transaction_amount, Plaid.transaction_amount, Razorpay.amount, Stripe.amount, UPI.txn_amount
- Rails: account_to_account_credit_transfer, card_network_iso8583, open_banking, psp_gateway, upi
### payment_currency
- Semantic family: payment:currency
- Members: ISO20022.currency, Plaid.currency, Razorpay.currency, Stripe.currency, UPI.currency
- Rails: account_to_account_credit_transfer, open_banking, psp_gateway, upi
### payment_identifier
- Semantic family: identifier:razorpay_payment
- Members: Razorpay.razorpay_payment_id
- Rails: psp_gateway
### payment_intent_identifier
- Semantic family: identifier:payment_intent
- Members: Stripe.payment_intent_id
- Rails: psp_gateway
### payment_method
- Semantic family: payment:method
- Members: Razorpay.method, Stripe.payment_method
- Rails: psp_gateway
### payment_status
- Semantic family: payment:status
- Members: Razorpay.status, Stripe.status, UPI.txn_status
- Rails: psp_gateway, upi
### processing_code
- Semantic family: iso8583:processing_code
- Members: Mastercard.processing_code
- Rails: card_network_iso8583
### requested_execution_date
- Semantic family: temporal:requested_execution_date
- Members: ISO20022.reqd_exctn_dt
- Rails: account_to_account_credit_transfer
### routing_number
- Semantic family: payment_account:routing_number
- Members: Plaid.routing_number
- Rails: open_banking
### system_trace_audit_number
- Semantic family: identifier:stan
- Members: Mastercard.system_trace_audit_number
- Rails: card_network_iso8583
### transaction_date
- Semantic family: temporal:transaction_date
- Members: Plaid.transaction_date
- Rails: open_banking
### transaction_identifier
- Semantic family: identifier:transaction
- Members: Plaid.transaction_id, UPI.txn_id
- Rails: open_banking, upi
### transaction_timestamp
- Semantic family: temporal:transaction_timestamp
- Members: UPI.txn_timestamp
- Rails: upi
### transmission_datetime
- Semantic family: temporal:transmission_datetime
- Members: Mastercard.transmission_datetime
- Rails: card_network_iso8583
### virtual_payment_address
- Semantic family: upi:vpa
- Members: Razorpay.vpa, UPI.vpa
- Rails: psp_gateway, upi

## Payload Compliance Summary
- ALL: ALLOW = 64
- INAmex: ALLOW = 8
- ISO20022: ALLOW = 8
- Mastercard: ALLOW = 8
- PPVisa: ALLOW = 8
- Plaid: ALLOW = 8
- Razorpay: ALLOW = 8
- Stripe: ALLOW = 8
- UPI: ALLOW = 8
- ALL: TOTAL = 64

## Normal Forms and Claim Support
- EENF: PASS — q95=0.0
- AANF: PASS — production_accepted_merges=37
- ECNF: PASS — candidate_deltas=0
- RRNF: PASS — rejected=14
- CMNF: PASS — rails modeled
- PONF: PASS — typed partitions enforced
- DBNF: PASS — version_drift: 41 nodes evaluated, 0 drifted, 0 forked
- C1: SUPPORTED — schema_attributes_ingested=61; canonical_node_count=41
- C2: REVISED — production_merge_precision=0.4
- C3: PARTIALLY_SUPPORTED — production_recall=0.75; discovery_recall=0.75
- C4: REVISED — cross_rail_merge_count=12; total_merges=37; leakage=0.32432432432432434
- C5: REVISED — {}
- C6: SUPPORTED — {"record_type": "timing", "candidate_generation_mode": "hnsw", "num_attributes": 61, "num_pairs_evaluated": 381, "num_candidates_considered": 381, "embedding_ms": 421.13179999997374, "index_build_ms": 5.244200000073761, "query_ms": 2.0346000000135973, "nf_validation_ms": null, "total_pipeline_ms": 428.4106000000611, "hnsw_M": 32, "hnsw_ef_search": 50, "top_k": 10, "speedup_vs_bruteforce_query": 19.916396343153373}
- C7: SUPPORTED — {"mode": "version_drift", "model_v1": "all-MiniLM-L6-v2", "model_v2": "all-MiniLM-L6-v2", "total_nodes_evaluated": 41, "drift_detected_count": 0, "fork_required_count": 0, "mean_drift_distance": 0.0690110766189974, "max_drift_distance": 0.08639490698465035, "tau_dbnf": 0.25, "drift_source": "simulated_perturbation", "ground_truth_eval": null, "delta_context": {"candidate_delta_count": 0, "missing_required_count": 0, "unexpected_field_count": 0}}
- C8: NOT_YET_SUPPORTED — {}

## Key Reviewer Takeaways
- Schema descriptors are authoritative contracts.
- Payloads are empirical evidence, not the schema source of truth.
- Identifier subtypes remain separated to avoid over-merging.
- DBNF version_drift is the paper claim; DBNF_MIGRATION is utility-only and not a paper claim.