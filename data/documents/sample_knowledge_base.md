# Acme Cloud — Customer Handbook

This is sample content so you can test the assistant end to end before adding
your own documents. Delete this file once you have real ones.

## Support hours and contact

Our support team is available Monday through Friday, 9:00 AM to 6:00 PM Pacific
Time. Weekend coverage is limited to critical incidents only. Enterprise
customers have 24/7 phone support through their dedicated success manager.

The general support email is support@acmecloud.example. Median first-response
time is under 2 hours during business hours.

## Plans and pricing

We offer three plans:

- **Starter** — $29 per month. Includes 5 seats, 50 GB of storage, and email
  support with a 24-hour response target.
- **Growth** — $99 per month. Includes 25 seats, 500 GB of storage, priority
  email and chat support, and single sign-on.
- **Enterprise** — custom pricing, starting around $1,200 per month. Unlimited
  seats, 5 TB of storage, 24/7 phone support, a dedicated success manager, and a
  99.95% uptime service level agreement.

Annual billing saves 20% compared to monthly billing. Plan changes take effect
immediately and we prorate the difference on the next invoice.

## Refunds and cancellation

New customers can request a full refund within 30 days of their first payment,
no questions asked. After 30 days, refunds are prorated for the unused portion
of the current billing period.

To cancel, open Settings, then Billing, then Cancel Subscription. Your data
remains available for export for 90 days after cancellation, after which it is
permanently deleted.

## Account access and password reset

To reset a password, click "Forgot password" on the sign-in page and enter the
email address on the account. The reset link is valid for 60 minutes. If the
email does not arrive within 5 minutes, check the spam folder or contact
support.

Accounts lock after 10 failed sign-in attempts and unlock automatically after 15
minutes. Administrators can unlock a locked account immediately from the Users
page.

Two-factor authentication is available on all plans and is required for
Enterprise accounts. We support authenticator apps and hardware security keys.
SMS-based two-factor was retired in March 2024.

## Data, security, and compliance

All data is encrypted in transit with TLS 1.3 and at rest with AES-256. We are
SOC 2 Type II certified and complete an independent penetration test annually.

Customer data is stored in the region selected at signup: United States, the
European Union (Frankfurt), or Asia Pacific (Singapore). Data does not leave the
selected region. We are GDPR compliant and will sign a data processing agreement
on request.

Backups run every 6 hours and are retained for 35 days. The recovery point
objective is 6 hours and the recovery time objective is 4 hours.

## Service level agreement

The Enterprise SLA guarantees 99.95% monthly uptime. If we fall below that, the
account receives a service credit: 10% of the monthly fee for uptime between
99.0% and 99.95%, 25% between 95.0% and 99.0%, and 50% below 95.0%.

Scheduled maintenance is announced at least 72 hours in advance and takes place
in a Sunday 02:00–06:00 window in the account's chosen region.

## Integrations

We integrate natively with Slack, Microsoft Teams, Jira, GitHub, Salesforce, and
Zapier. The REST API is rate limited to 1,000 requests per minute on Growth and
10,000 requests per minute on Enterprise. Webhooks retry failed deliveries five
times with exponential backoff over roughly 30 minutes.

## Onboarding

Growth and Enterprise customers get a guided onboarding session. Typical
onboarding takes 2 weeks for Growth and 4 to 6 weeks for Enterprise, including
data migration and single sign-on configuration. Migration tooling supports
imports from CSV, S3, and our legacy v1 API.
