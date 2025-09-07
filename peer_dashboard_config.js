module.exports = {
  apps: [{
    name: 'peer-dashboard',
    script: 'peer_dashboard.py',
    interpreter: 'python',
    env: {
      NODE_ENV: 'development'
    },
    log_file: './logs/peer-dashboard.log',
    out_file: './logs/peer-dashboard-out.log',
    error_file: './logs/peer-dashboard-error.log',
    restart_delay: 3000,
    max_restarts: 10
  }]
}