// ==================================================================
// This is to avoid getting warnings when reading local json files
// ==================================================================
$.ajaxSetup({
  beforeSend: function(xhr) {
    if (xhr.overrideMimeType) {
      xhr.overrideMimeType("application/json");
    }
  }
});


// ==================================================================
// Mapbox access token
// ==================================================================
var mapboxAccessToken = 'pk.eyJ1IjoiamFzaW5naCIsImEiOiJjamhpNThlZ2UyMTZyM2FwM3g4Z2Q2Nnh1In0.sakrRrBxJrAhZxJHiZUUmw'

// ==================================================================
// Helper variables and functions
// ==================================================================
var sec_to_millisec = 1000
var millisec_to_sec = 1 / sec_to_millisec

function isEmpty(obj) {
  for (var key in obj) {
    if (obj.hasOwnProperty(key))
      return false;
  }
  return true;
}

function linearMap(elems, offset, scale) {
  var out = [];
  for (var i = 0; i < elems.length; i++) {
    out.push((elems[i] - offset) * scale)
  }
  return out
}

function scatter(x, y, name, xaxis, yaxis, text = '') {
  return {
    type: 'scatter',
    x: x,
    y: y,
    name: name,
    xaxis: xaxis,
    yaxis: yaxis,
    text: text
  }
}


function scattergl(x, y, name, text = '') {
  return {
    type: 'scattergl',
    x: x,
    y: y,
    name: name,
    mode: 'markers',
    text: text
  }
}


function scattermapbox(lat, lon, name, text) {
  return {
    type: 'scattermapbox',
    lat: lat,
    lon: lon,
    name: name,
    mode: 'markers+lines',
    marker: {
      size: 3
    },
    text: text
  }
}


function three_subplot_layout(title, height=900) {
  var layout = {}
  layout['height'] = height
  layout['title'] = title
  layout['xaxis1'] = {
    domain: [0.2, 0.8],
    anchor: 'y1'
  }
  layout['xaxis2'] = {
    domain: [0.2, 0.8],
    anchor: 'y2'
  }
  layout['xaxis3'] = {
    domain: [0.2, 0.8],
    anchor: 'y3'
  }
  layout['yaxis1'] = {
    domain: [0.7, 1.0],
    anchor: 'x1'
  }
  layout['yaxis2'] = {
    domain: [0.35, 0.65],
    anchor: 'x2'
  }
  layout['yaxis3'] = {
    domain: [0.0, 0.3],
    anchor: 'x3'
  }
  return layout
}


function estimate_yaw_from_pose_xy(t, x, y) {
  var K = 100 // (i - K, i + K) = 2 second window
  var MIN_DIST = 1 * 2 * K / 100 // min speed = 1 m/s
  var valid_t = []
  var valid_yaw = []
  for (var i = 0; i < t.length; i++) {
    if (i - K >= 0 && i + K < t.length) {
      var dy = y[i + K] - y[i - K]
      var dx = x[i + K] - x[i - K]
      if (Math.hypot(dy, dx) > MIN_DIST) {
        valid_t.push(t[i])
        valid_yaw.push(Math.atan2(dy, dx))
      }
    }
  }
  return [valid_t, valid_yaw]
}


function estimate_yaw_from_pose_vel(t, vx, vy) {
  var MIN_SPEED = 1  // m/s
  var valid_t = []
  var valid_yaw = []
  for (var i = 0; i < t.length; i++) {
    if (Math.hypot(vy[i], vx[i]) > MIN_SPEED) {
      valid_t.push(t[i])
      valid_yaw.push(Math.atan2(vy[i], vx[i]))
    }
  }
  return [valid_t, valid_yaw]
}


function zero_to_360_degrees(yaw) {
  var out = []
  for (var i = 0; i < yaw.length; i++) {
    var current_yaw = yaw[i];
    while (current_yaw < 0) {
      current_yaw += 2 * Math.PI
    }
    while (current_yaw >= 2 * Math.PI) {
      current_yaw -= 2 * Math.PI
    }
    out.push(current_yaw*180.0/Math.PI)
  }
  return out
}

$.ajaxSetup({
  async: false
});

var pose_names = ['ins_pose', 'fusion_pose', 'pose', 'lidar_pose', 'loam_pose']

// just some default values, will get updated based on lat-lon data below
var refLat = 37.0
var refLon = -122.0
var refTime = 0


// ============================================================
// Update layouts and plot everything
// ============================================================

// ==================================================================
// Read pose data for satellite view, rpy plot, xy variance plot
// ==================================================================
var lat_lon_plot_data = []
var x_y_plot_data = []
var vel_plot_data = []
var acc_plot_data = []
var rpy_plot_data = []
var xyz_var_plot_data = []
var gps_imu_cov_plot_data = []

pose_names.forEach(function(pose_name) {
  $.getJSON('json/' + pose_name + '.json').done(function(pose_data) {
    console.log("Reading pose data: " + pose_name)
    if (isEmpty(pose_data)) {
      console.log("Skipping " + pose_name + " it's missing")
    } else {
      if (refTime == 0 && pose_data.t.length > 0) {
        refTime = pose_data.t[0]
      }

      if (pose_data.lat.length > 0 && pose_data.lon.length > 0) {
        refLat = pose_data.lat[0]
        refLon = pose_data.lon[0]
      }

      // Satellite View: lat-lon/xy plots
      var relativeTime = linearMap(pose_data.t, refTime, millisec_to_sec)
      var text_to_show = []
      for (var i = 0; i < pose_data.t.length; i++) {
        var text_str = 't=' + relativeTime[i] + ' ts=' + pose_data.t[i]
        if (pose_name === "lidar_pose" && pose_data.hasOwnProperty('xy_status')) {
          text_str += ' xy_status=' + pose_data.xy_status[i]
        } else if (pose_name === "ins_pose") {
          if (pose_data.hasOwnProperty('status')) {
            text_str += ' status=' + pose_data.status[i]
          }
          if (pose_data.hasOwnProperty('pos_type')) {
            text_str += ' pos_type=' + pose_data.pos_type[i]
          }
        }
        text_to_show.push(text_str)
      }
      var loc_pose_lat_lon_plot = scattermapbox(pose_data.lat, pose_data.lon, pose_name, text_to_show)
      lat_lon_plot_data.push(loc_pose_lat_lon_plot)

      var loc_pose_x_y_plot = scattergl(pose_data.x, pose_data.y, pose_name, text_to_show)
      x_y_plot_data.push(loc_pose_x_y_plot)

      if(!pose_name.includes('lidar_pose') && !pose_name.includes('slam_pose')) {
        // velocity plot
        vel_plot_data.push(scatter(relativeTime, pose_data.vel_x, 'vel_x_' + pose_name, 'x1', 'y1'))
        vel_plot_data.push(scatter(relativeTime, pose_data.vel_y, 'vel_y_' + pose_name, 'x2', 'y2'))
        vel_plot_data.push(scatter(relativeTime, pose_data.vel_z, 'vel_z_' + pose_name, 'x3', 'y3'))

        // acceleration plot
        acc_plot_data.push(scatter(relativeTime, pose_data.acc_right, 'acc_right_' + pose_name, 'x1', 'y1'))
        acc_plot_data.push(scatter(relativeTime, pose_data.acc_forward, 'acc_forward_' + pose_name, 'x2', 'y2'))
        acc_plot_data.push(scatter(relativeTime, pose_data.acc_up, 'acc_up_' + pose_name, 'x3', 'y3'))
      }

      // rpy plot
      rpy_plot_data.push(scatter(relativeTime, pose_data.roll, 'roll_' + pose_name, 'x1', 'y1'))
      rpy_plot_data.push(scatter(relativeTime, pose_data.pitch, 'pitch_' + pose_name, 'x2', 'y2'))
      rpy_plot_data.push(scatter(relativeTime, zero_to_360_degrees(pose_data.yaw), 'yaw_' + pose_name, 'x3', 'y3'))

      // special ins_pose-fitted yaw-estimate.
      // sometimes ins_pose topic is missing, use pose in that case.
      if (pose_name.includes('ins_pose') || pose_name == 'pose') {
        var valid_time_yaw = estimate_yaw_from_pose_xy(relativeTime, pose_data.x, pose_data.y)
        rpy_plot_data.push(scatter(valid_time_yaw[0], zero_to_360_degrees(valid_time_yaw[1]), 'yaw_from_' + pose_name + '_xy', 'x3', 'y3'))
      }

      // xy variance plot
      xyz_var_plot_data.push(scatter(relativeTime, pose_data.x_var, 'x_var_' + pose_name, 'x1', 'y1'))
      xyz_var_plot_data.push(scatter(relativeTime, pose_data.y_var, 'y_var_' + pose_name, 'x2', 'y2'))
      xyz_var_plot_data.push(scatter(relativeTime, pose_data.z_var, 'z_var_' + pose_name, 'x3', 'y3'))

      // gps_imu covariance plot.
      if (pose_name.includes('ins_pose')) {
        if (pose_data.hasOwnProperty('lat_cov')) {
          gps_imu_cov_plot_data.push(scatter(relativeTime, pose_data.lat_cov, 'lat_cov', 'x1', 'y1', text_to_show))
        }
        if (pose_data.hasOwnProperty('long_cov')) {
          gps_imu_cov_plot_data.push(scatter(relativeTime, pose_data.long_cov, 'long_cov', 'x2', 'y2', text_to_show))
        }
        if (pose_data.hasOwnProperty('alt_cov')) {
          gps_imu_cov_plot_data.push(scatter(relativeTime, pose_data.alt_cov, 'alt_cov', 'x3', 'y3', text_to_show))
        }
      }
    }
  })
})

var lat_lon_plot_layout = {}
var x_y_plot_layout = {}

var POSE_PLOT_HEIGHT = 1200
var POSE_PLOT_WIDTH = 1800
// lat-lon pose
lat_lon_plot_layout['height'] = POSE_PLOT_HEIGHT
lat_lon_plot_layout['title'] = 'Lat-Lon Pose'
lat_lon_plot_layout['hover_mode'] = 'closest'
lat_lon_plot_layout['mapbox'] = {
  accesstoken: mapboxAccessToken,
  bearing: 0,
  center: {
    lat: refLat,
    lon: refLon
  },
  zoom: 20,
  style: 'satellite-streets',
  domain: {
    x: [0, 1],
    y: [0, 1]
  }
}

Plotly.newPlot('lat_lon_div', lat_lon_plot_data, lat_lon_plot_layout, {
  showSendToCloud: true
});

// x-y pose
x_y_plot_layout['height'] = POSE_PLOT_HEIGHT
x_y_plot_layout['width'] = POSE_PLOT_WIDTH
x_y_plot_layout['title'] = 'X-Y Pose'

// velocity
Plotly.newPlot('vel_curve_div', vel_plot_data, three_subplot_layout('Velocity (m/s)'), {
  showSendToCloud: true
});

// acceleration
Plotly.newPlot('acc_curve_div', acc_plot_data, three_subplot_layout('Acceleration (m/s^2)'), {
  showSendToCloud: true
});

// rpy
Plotly.newPlot('rpy_curve_div', rpy_plot_data, three_subplot_layout('RPY (radians)'), {
  showSendToCloud: true
});

// xyz variance
Plotly.newPlot('xyz_var_curve_div', xyz_var_plot_data, three_subplot_layout('XYZ Variance (m/s^2)'), {
  showSendToCloud: true
});

// gps_imu covariance
Plotly.newPlot('gps_imu_curve_div', gps_imu_cov_plot_data, three_subplot_layout('GpsImu Position Covariance(m/s^2)'), {
  showSendToCloud: true
});

// ============================================================
// Lat-lon/XY toggle button click handler
// ============================================================
var toggle = true;
$("button").click(function() {
  if (toggle) {
    Plotly.newPlot('lat_lon_div', x_y_plot_data, x_y_plot_layout, {
      showSendToCloud: true
    });
  } else {
    Plotly.newPlot('lat_lon_div', lat_lon_plot_data, lat_lon_plot_layout, {
      showSendToCloud: true
    });
  }
  toggle = !toggle;
})
