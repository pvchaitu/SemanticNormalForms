# SDNF v15 Fixed Payment Payloads — 8 per Payment Type

This bundle contains 64 generated payload files: 8 payloads each for INAmex, PPVisa, Mastercard, ISO20022, Plaid, Razorpay, Stripe, and UPI.

## Payload generation policy

- Payload fields are restricted to schema-declared attributes or explicit aliases from the corresponding `*.schema.json` files.
- v14 payload-only fields that produced unexpected-field onboarding noise were removed unless the corresponding schema declared the attribute or alias.
- Mastercard `transmission_datetime` is generated as a 10-digit synthetic ISO8583-style value to satisfy the schema pattern `^[0-9]{10}$`.
- Razorpay and Stripe `amount` values are generated as integers in minor currency units to match their schema descriptors.
- The folder structure is compatible with the experiment layout: `payloads/payment/<PaymentType>/*.json`.

## Recommended usage

Unzip this archive at the experiment project root so that the folder `payloads/payment` is available to the SDNF experiment.
