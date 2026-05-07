# SDNF Synthetic Payment Payloads v16

This package contains 6 synthetic payload files for each payment type:

- INAmex
- PPVisa
- Mastercard
- ISO20022
- Plaid
- Razorpay
- Stripe
- UPI

Total payload files: 48.

## Intended usage

Use these files as payload evidence for the SDNF experiment:

```text
payloads/payment/<payment_type>/*.json
        ↓
payload profiler
        ↓
provisional schema inference
        ↓
compare against schema descriptors
        ↓
SDNF gates
        ↓
Master Payment SRS lineage and compliance regression
```

## Important safety and research notes

All payload values are synthetic. Card-like, account-like, CVV-like, VPA-like, and transaction-like values are generated only to support value-shape testing. They are not real payment credentials and must not be used for real payment processing.

Payloads should be used as ECNF/value evidence and regression data. They should not be treated as authoritative provider specifications.

## Field presence logic

With 6 payloads per type:

```text
6/6 fields  -> required candidate
4-5/6       -> conditional or strong optional candidate
2-3/6       -> optional or method-specific candidate
1/6         -> outlier / low-confidence candidate
```

## Design choices

- ISO20022 payloads include `dbtr_acct` and `cdtr_acct` to avoid ambiguous generic `acct_num` role collapse.
- Razorpay payloads separate PSP fields from card-instrument fields; card fields appear only for card-method examples.
- Stripe payloads are PaymentIntent-like and include raw card aliases only in selected examples to test payload-first inference and schema alignment.
- UPI payloads preserve payer/payee roles for RRNF testing.
- Identifier fields are intentionally scoped by provider/entity to test PONF and prevent over-merging.
