"use strict";
/**
 * Spherical Elementary Current Systems (SECS) implementation in TypeScript
 * Based on Amm & Viljanen's paper: "Ionospheric disturbance magnetic field continuation
 * from the ground to the ionosphere using spherical elementary current systems."
 */
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.processMagneticFieldData = processMagneticFieldData;
// Earth's radius in meters
var R_EARTH = 6371e3;
// Magnetic permeability constant
var MU0 = 4 * Math.PI * 1e-7;
/**
 * Calculates the angular distance between two sets of lat/lon points
 * @param latlon1 First set of points [lat, lon]
 * @param latlon2 Second set of points [lat, lon]
 * @returns Matrix of angular distances in radians
 */
function calcAngularDistance(latlon1, latlon2) {
    var result = [];
    for (var i = 0; i < latlon1.length; i++) {
        var row = [];
        var lat1Rad = (latlon1[i][0] * Math.PI) / 180;
        var lon1Rad = (latlon1[i][1] * Math.PI) / 180;
        for (var j = 0; j < latlon2.length; j++) {
            var lat2Rad = (latlon2[j][0] * Math.PI) / 180;
            var lon2Rad = (latlon2[j][1] * Math.PI) / 180;
            var dlon = lon2Rad - lon1Rad;
            // Theta = angular distance between two points
            var theta = Math.acos(Math.sin(lat1Rad) * Math.sin(lat2Rad) +
                Math.cos(lat1Rad) * Math.cos(lat2Rad) * Math.cos(dlon));
            row.push(theta);
        }
        result.push(row);
    }
    return result;
}
/**
 * Calculates the bearing (direction) between two sets of lat/lon points
 * @param latlon1 First set of points [lat, lon]
 * @param latlon2 Second set of points [lat, lon]
 * @returns Matrix of bearings in radians
 */
function calcBearing(latlon1, latlon2) {
    var result = [];
    for (var i = 0; i < latlon1.length; i++) {
        var row = [];
        var lat1Rad = (latlon1[i][0] * Math.PI) / 180;
        var lon1Rad = (latlon1[i][1] * Math.PI) / 180;
        for (var j = 0; j < latlon2.length; j++) {
            var lat2Rad = (latlon2[j][0] * Math.PI) / 180;
            var lon2Rad = (latlon2[j][1] * Math.PI) / 180;
            var dlon = lon2Rad - lon1Rad;
            // Alpha = bearing from point1 to point2
            var alpha = Math.PI / 2 - Math.atan2(Math.sin(dlon) * Math.cos(lat2Rad), Math.cos(lat1Rad) * Math.sin(lat2Rad) -
                Math.sin(lat1Rad) * Math.cos(lat2Rad) * Math.cos(dlon));
            row.push(alpha);
        }
        result.push(row);
    }
    return result;
}
/**
 * Calculates the Transfer matrix for divergence-free SECS
 * @param obsLoc Observation locations
 * @param secLoc SEC locations
 * @returns The T transfer matrix
 */
function T_df(obsLoc, secLoc) {
    var nobs = obsLoc.length;
    var nsec = secLoc.length;
    // Convert to lat/lon arrays for distance calculations
    var obsLatLon = obsLoc.map(function (p) { return [p.lat, p.lon]; });
    var secLatLon = secLoc.map(function (p) { return [p.lat, p.lon]; });
    // Calculate angular distances and bearings
    var theta = calcAngularDistance(obsLatLon, secLatLon);
    var alpha = calcBearing(obsLatLon, secLatLon);
    // Initialize transfer matrix
    var T = Array(nobs).fill(0).map(function () {
        return Array(3).fill(0).map(function () { return Array(nsec).fill(0); });
    });
    // Calculate transfer function for each observation-SEC pair
    for (var i = 0; i < nobs; i++) {
        var obsR = obsLoc[i].r;
        for (var j = 0; j < nsec; j++) {
            var secR = secLoc[j].r;
            var x = obsR / secR;
            var sinTheta = Math.sin(theta[i][j]);
            var cosTheta = Math.cos(theta[i][j]);
            // Factor used in calculations
            var factor = 1.0 / Math.sqrt(1 - 2 * x * cosTheta + x * x);
            // Radial component - Amm & Viljanen: Equation 9
            var Br = MU0 / (4 * Math.PI * obsR) * (factor - 1);
            // Theta component - Amm & Viljanen: Equation 10
            var Btheta = -MU0 / (4 * Math.PI * obsR) * (factor * (x - cosTheta) + cosTheta);
            if (sinTheta !== 0) {
                Btheta /= sinTheta;
            }
            else {
                Btheta = 0;
            }
            // Check if SEC is below observation
            if (secR < obsR) {
                // Flipped calculation for SECs below observations
                var x_1 = secR / obsR;
                // Amm & Viljanen: Equation A.7
                Br = MU0 * x_1 / (4 * Math.PI * obsR) *
                    (1.0 / Math.sqrt(1 - 2 * x_1 * cosTheta + x_1 * x_1) - 1);
                // Amm & Viljanen: Equation A.8
                Btheta = -MU0 / (4 * Math.PI * obsR) * ((obsR - secR * cosTheta) /
                    Math.sqrt(obsR * obsR - 2 * obsR * secR * cosTheta + secR * secR) - 1);
                if (sinTheta !== 0) {
                    Btheta /= sinTheta;
                }
                else {
                    Btheta = 0;
                }
            }
            // Transform to Cartesian coordinates
            T[i][0][j] = -Btheta * Math.sin(alpha[i][j]); // Bx
            T[i][1][j] = -Btheta * Math.cos(alpha[i][j]); // By
            T[i][2][j] = -Br; // Bz
        }
    }
    return T;
}
/**
 * Creates a regular grid of SEC poles
 * @returns Array of SEC pole locations
 */
function createSECGrid() {
    // Create a grid similar to the one in the original code
    var lat = linspace(45, 85, 50);
    var lon = linspace(-180, 179, 50);
    var r = R_EARTH + 110000; // Constant value for radius
    var secLocs = [];
    // Create grid points
    for (var _i = 0, lat_1 = lat; _i < lat_1.length; _i++) {
        var lt = lat_1[_i];
        for (var _a = 0, lon_1 = lon; _a < lon_1.length; _a++) {
            var ln = lon_1[_a];
            secLocs.push({
                lat: lt,
                lon: ln,
                r: r
            });
        }
    }
    return secLocs;
}
/**
 * Utility function to create evenly spaced values
 * @param start Start value
 * @param end End value
 * @param n Number of values
 * @returns Array of evenly spaced values
 */
function linspace(start, end, n) {
    if (n === void 0) { n = 50; }
    var result = [];
    var step = (end - start) / (n - 1);
    for (var i = 0; i < n; i++) {
        result.push(start + step * i);
    }
    return result;
}
/**
 * Solves a linear system using SVD (Singular Value Decomposition)
 * Simplified implementation for demonstration purposes
 * @param A Matrix A
 * @param b Vector b
 * @param epsilon Regularization parameter
 * @returns Solution vector x
 */
function solveSVD(A, b, epsilon) {
    // Basic implementation of SVD solving
    // In a real implementation, use a proper linear algebra library
    if (epsilon === void 0) { epsilon = 0.1; }
    // Transpose for calculations
    var AT = transposeMatrix(A);
    // Calculate AT*A
    var ATA = multiplyMatrices(AT, A);
    // Add regularization to diagonal
    var maxDiag = Math.max.apply(Math, ATA.map(function (row, i) { return row[i]; }));
    var reg = epsilon * maxDiag;
    for (var i = 0; i < ATA.length; i++) {
        ATA[i][i] += reg;
    }
    // Calculate inverse of (AT*A)
    var ATAinv = invertMatrix(ATA);
    // Calculate AT*b
    var ATb = multiplyMatrixVector(AT, b);
    // Calculate (AT*A)^-1 * (AT*b)
    return multiplyMatrixVector(ATAinv, ATb);
}
/**
 * Simple matrix transpose
 * @param matrix Input matrix
 * @returns Transposed matrix
 */
function transposeMatrix(matrix) {
    var rows = matrix.length;
    var cols = matrix[0].length;
    var result = Array(cols).fill(0).map(function () { return Array(rows).fill(0); });
    for (var i = 0; i < rows; i++) {
        for (var j = 0; j < cols; j++) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}
/**
 * Simple matrix multiplication
 * @param A First matrix
 * @param B Second matrix
 * @returns Result matrix
 */
function multiplyMatrices(A, B) {
    var rowsA = A.length;
    var colsA = A[0].length;
    var colsB = B[0].length;
    var result = Array(rowsA).fill(0).map(function () { return Array(colsB).fill(0); });
    for (var i = 0; i < rowsA; i++) {
        for (var j = 0; j < colsB; j++) {
            for (var k = 0; k < colsA; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}
/**
 * Matrix-vector multiplication
 * @param A Matrix
 * @param v Vector
 * @returns Result vector
 */
function multiplyMatrixVector(A, v) {
    var rows = A.length;
    var cols = A[0].length;
    var result = Array(rows).fill(0);
    for (var i = 0; i < rows; i++) {
        for (var j = 0; j < cols; j++) {
            result[i] += A[i][j] * v[j];
        }
    }
    return result;
}
/**
 * Simple matrix inversion (for demonstration only)
 * In a real implementation, use a proper linear algebra library
 * @param A Input matrix
 * @returns Inverted matrix
 */
function invertMatrix(A) {
    var _a, _b;
    // This is a simplified implementation that only works for well-conditioned matrices
    // For real applications, use a proper linear algebra library
    var n = A.length;
    var result = Array(n).fill(0).map(function () { return Array(n).fill(0); });
    var temp = [];
    // Copy A and create identity matrix
    for (var i = 0; i < n; i++) {
        temp[i] = __spreadArray([], A[i], true);
        result[i][i] = 1;
    }
    // Gauss-Jordan elimination
    for (var i = 0; i < n; i++) {
        // Find pivot
        var max = Math.abs(temp[i][i]);
        var maxRow = i;
        for (var j = i + 1; j < n; j++) {
            if (Math.abs(temp[j][i]) > max) {
                max = Math.abs(temp[j][i]);
                maxRow = j;
            }
        }
        // Swap rows
        if (maxRow !== i) {
            _a = [temp[maxRow], temp[i]], temp[i] = _a[0], temp[maxRow] = _a[1];
            _b = [result[maxRow], result[i]], result[i] = _b[0], result[maxRow] = _b[1];
        }
        // Scale row
        var pivot = temp[i][i];
        for (var j = 0; j < n; j++) {
            temp[i][j] /= pivot;
            result[i][j] /= pivot;
        }
        // Eliminate other rows
        for (var j = 0; j < n; j++) {
            if (j !== i) {
                var factor = temp[j][i];
                for (var k = 0; k < n; k++) {
                    temp[j][k] -= factor * temp[i][k];
                    result[j][k] -= factor * result[i][k];
                }
            }
        }
    }
    return result;
}
/**
 * Main function to interpolate data using SECS method
 * @param observations Array of observation points
 * @returns Array of interpolated points
 */
function interpolateWithSECS(observations) {
    // Create SEC grid
    var secLocations = createSECGrid();
    // Create observation locations array with proper structure
    var obsLocations = observations.map(function (obs) { return ({
        lat: obs.y,
        lon: obs.x,
        r: R_EARTH // Assume observations are at Earth's surface
    }); });
    // Calculate transfer matrix
    var T = T_df(obsLocations, secLocations);
    // Prepare the system for solving
    // Flatten the transfer matrix for calculations
    var flatT = [];
    for (var i = 0; i < obsLocations.length; i++) {
        // We only use Bx and By components (i and j in obs)
        flatT.push(T[i][0]); // Bx row
        flatT.push(T[i][1]); // By row
    }
    // Flatten the observation vector
    var flatB = [];
    for (var _i = 0, observations_1 = observations; _i < observations_1.length; _i++) {
        var obs = observations_1[_i];
        flatB.push(obs.i); // Bx component
        flatB.push(obs.j); // By component
    }
    // Solve the system to get SEC amplitudes
    var secAmps = solveSVD(flatT, flatB, 0.1);
    // Create prediction grid
    var latPred = linspace(45, 85, 37);
    var lonPred = linspace(-180, 179, 130);
    var predLocations = [];
    for (var _a = 0, latPred_1 = latPred; _a < latPred_1.length; _a++) {
        var lat = latPred_1[_a];
        for (var _b = 0, lonPred_1 = lonPred; _b < lonPred_1.length; _b++) {
            var lon = lonPred_1[_b];
            predLocations.push({
                lat: lat,
                lon: lon,
                r: R_EARTH
            });
        }
    }
    // Calculate transfer matrix for prediction locations
    var T_pred = T_df(predLocations, secLocations);
    // Calculate predicted field values
    var result = [];
    for (var i = 0; i < predLocations.length; i++) {
        var Bx = 0;
        var By = 0;
        // Apply SEC amplitudes to transfer matrix
        for (var j = 0; j < secLocations.length; j++) {
            Bx += T_pred[i][0][j] * secAmps[j];
            By += T_pred[i][1][j] * secAmps[j];
        }
        result.push({
            lon: Math.round(predLocations[i].lon * 100) / 100, // Round to 2 decimal places
            lat: Math.round(predLocations[i].lat * 100) / 100, // Round to 2 decimal places
            i: Math.round(Bx), // Round to integer
            j: Math.round(By) // Round to integer
        });
    }
    return result;
}
/**
 * Main entry function for SECS interpolation
 * @param data Array of observation points with x (longitude), y (latitude), i and j components in nT
 * @returns Array of interpolated points
 */
function processMagneticFieldData(data) {
    // Validate input data
    if (!data || !data.x || !data.y || !data.i || !data.j) {
        throw new Error('Invalid input data structure');
    }
    if (data.x.length !== data.y.length ||
        data.x.length !== data.i.length ||
        data.x.length !== data.j.length) {
        throw new Error('Input arrays must have the same length');
    }
    // Convert input data to array of observation points
    var observations = data.x.map(function (x, index) { return ({
        x: x,
        y: data.y[index],
        i: data.i[index],
        j: data.j[index]
    }); });
    // Process with SECS interpolation
    return interpolateWithSECS(observations);
}
// Example usage:
// const result = processMagneticFieldData({
//   x: [11.95, 15.82, 25.01, /* ... */],
//   y: [78.92, 78.2, 76.51, /* ... */],
//   i: [-51, -27, -7, /* ... */],
//   j: [-36, -44, -39, /* ... */]
// });
// Example usage:
// const result = processMagneticFieldData({
//   x: [11.95, 15.82, 25.01, /* ... */],
//   y: [78.92, 78.2, 76.51, /* ... */],
//   i: [-51, -27, -7, /* ... */],
//   j: [-36, -44, -39, /* ... */]
// });
//
// Example usage:
var result = processMagneticFieldData({
    x: [
        11.95, 15.82, 25.01, 19.2, 351.3,
        25.79, 12.1, 22.22, 27.01, 18.94,
        26.63, 23.7, 16.03, 16.98, 20.77,
        27.29, 23.53, 24.08, 12.5, 26.25,
        10.98, 27.23, 30.97, 26.6, 9.11,
        4.84, 10.75, 24.65, 5.24, 26.46,
        11.67, 290.77, 321.7, 341.37, 307.87,
        306.47, 309.28, 314.56, 307.1, 254.763,
        203.378, 212.14, 242.878, 199.538, 224.675,
        297.647, 264, 262.9, 255, 274.1,
        265.9, 291.5, 246.7, 284.5, 265.1,
        307.3, 236.6, -147.447, -141.205, -149.592
    ],
    y: [
        78.92, 78.2, 76.51, 74.5, 70.9, 71.09,
        67.53, 70.54, 69.76, 69.66, 67.37, 69.46,
        69.3, 66.4, 69.06, 68.56, 68.02, 66.9,
        66.11, 65.54, 64.94, 64.52, 62.77, 62.25,
        62.07, 61.08, 60.21, 60.5, 59.21, 58.26,
        55.62, 77.47, 72.3, 76.77, 70.68, 69.25,
        67.02, 61.16, 65.42, 40.137, 71.322, 64.874,
        48.265, 55.348, 57.058, 82.497, 64.3, 49.6,
        69.2, 80, 58.8, 63.8, 54.6, 45.4,
        74.7, 47.6, 48.6, 65.136, 64.786, 68.627
    ],
    i: [
        -51, -27, -7, 117, 103, 103, 82, 97, 83,
        96, 72, 79, 99, 63, 87, 66, 79, 66,
        58, 48, 41, 36, 25, 3, 11, -2, -2,
        6, -9, 2, -11, -10, 36, -86, 179, -14,
        309, -2, 52, -12, -74, -102, -2, 13, -13,
        -52, -109, -12, 12, -6, -35, -8, -19, -70,
        63, -69, -20, -250, -164, -140
    ],
    j: [
        -36, -44, -39, -38, -17, -40, -34, -42, -56,
        -39, -31, -53, -39, -36, -40, -52, -29, -28,
        -29, -22, -27, -25, -21, -33, -31, -32, -33,
        -24, -37, -23, -32, -107, -75, -21, 116, -88,
        -178, -2, 85, 48, -13, -49, 29, -2, 11,
        97, 45, 56, 51, 136, 35, 70, 37, 47,
        103, 22, 15, -146, 19, -41
    ]
});
var findScore = function (value) {
    // 0 0 
    // 50 2
    // 100 3
    // 200 4
    // 800 10
    var f = function (x) {
        var numerator = -0.05732817 - 22.81964;
        var denominator = 1 + Math.pow(x / 1055.17, 0.8849212);
        return 22.81964 + numerator / denominator;
    };
    return Math.min(10, f(value));
};
// 0 0.3
// 2200 1
// 2700 1
// 3300 0.5
// 4000 0.2
var ovalPonderation = function (x) {
    if (x >= 0 && x <= 2200) {
        return -2.0790e-11 * Math.pow(x, 3)
            - 7.6000e-67 * Math.pow(x, 2)
            + 5.5517e-4 * x
            + 0.0000;
    }
    else if (x > 2200 && x <= 2700) {
        return -7.3879e-10 * Math.pow(x, 3)
            + 4.7388e-6 * Math.pow(x, 2)
            - 9.8701e-3 * x
            + 7.6452;
    }
    else if (x > 2700 && x <= 3300) {
        return 9.7750e-10 * Math.pow(x, 3)
            - 9.1631e-6 * Math.pow(x, 2)
            + 2.7665e-2 * x
            - 2.6136e1;
    }
    else if (x > 3300 && x <= 5000) {
        return -1.0080e-10 * Math.pow(x, 3)
            + 1.5121e-6 * Math.pow(x, 2)
            - 7.5632e-3 * x
            + 1.2615e1;
    }
    else {
        return 0.3;
    }
};
var haversineDistance = function (lat1, lon1, lat2, lon2) {
    var R = 6371; // Earth's radius in km
    var toRad = function (deg) { return deg * (Math.PI / 180); };
    var dLat = toRad(lat2 - lat1);
    var dLon = toRad(lon2 - lon1);
    var a = Math.pow(Math.sin(dLat / 2), 2) +
        Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
            Math.pow(Math.sin(dLon / 2), 2);
    var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c; // Distance in km
};
// 80.9N 72.6W
var geomagneticNorthPole = [-72.6, 80.9];
console.log(JSON.stringify(result.map(function (v) {
    var x = findScore(Math.abs(v.i));
    var weight = Number(ovalPonderation(haversineDistance(geomagneticNorthPole[1], geomagneticNorthPole[0], v.lat, v.lon)));
    var derivativeScore = function (d) { return (d < 10 ? 0 : d / 10); };
    var score = weight * x;
    return {
        lon: v.lon,
        lat: v.lat,
        score: Number(Math.max(score).toFixed(3)),
    };
})
    .filter(function (v) { return v.score > 0; })));
