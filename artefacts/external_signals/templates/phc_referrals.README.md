# C1. Primary Health Care (PHC) referral counts — INTERNAL

## Source
Steve Biko Hospital administration / Tshwane Health District.
Requires data sharing agreement.

## Why
ED is the overflow buffer for surrounding clinics. Higher upstream PHC
referral volume on day D directly increases ED arrivals D+0 and D+1.

## Schema
- referrals_in_count            count of formal referrals to ED that day
- referrals_in_acuity_mean      mean acuity score of those referrals
- self_referrals_count          count of walk-ins (non-referred)
- referral_top_clinic           categorical, top contributor clinic name
