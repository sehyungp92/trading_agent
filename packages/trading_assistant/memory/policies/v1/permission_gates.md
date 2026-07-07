# Permission Gates — v1

## Auto (no approval needed)
- open_github_issue
- create_draft_pr
- add_logging
- add_tests
- update_documentation
- generate_report

Allowed paths: `docs/*`, `tests/*`, `*.md`

## Requires Approval (Telegram confirmation)
- merge_pr
- change_trading_logic
- change_risk_parameters
- change_position_sizing
- modify_filters
- update_strategy_config
- portfolio_allocation — family weight rebalancing (max 15% change per cycle)
- portfolio_coordination — add/modify coordination signals between strategies

Restricted paths: `strategies/*`, `risk/*`, `execution/*`, `sizing/*`, `filters/*`, `config/trading_*.yaml`, `config/portfolio.yaml:family_allocations`, `config/portfolio.yaml:coordination`

## Requires Double Approval (confirm twice with reason)
- change_api_keys
- modify_deployment
- change_kill_switch
- modify_exchange_connectivity
- change_permission_gates
- portfolio_risk_cap — heat_cap_R and daily/weekly stop level changes
- portfolio_drawdown_tier — drawdown tier multiplier changes (tightening only)

Restricted paths: `deploy/*`, `infra/*`, `.env*`, `keys/*`, `kill_switch*`, `memory/policies/*`, `skills/ground_truth_computer.py`, `config/portfolio.yaml:heat_cap_R`, `config/portfolio.yaml:drawdown_tiers`, `config/portfolio.yaml:daily_stop_R`
