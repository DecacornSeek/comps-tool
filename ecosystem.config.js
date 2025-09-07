module.exports = {
  apps: [{
    name: 'stock-dashboard',
    script: 'python',
    args: 'dashboard.py',
    cwd: '/home/user/webapp',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'production',
      PORT: 8050
    },
    log_file: '/home/user/webapp/logs/dashboard.log',
    out_file: '/home/user/webapp/logs/dashboard-out.log',
    error_file: '/home/user/webapp/logs/dashboard-error.log',
    time: true
  }]
};