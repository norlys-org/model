import * as d3 from 'd3'

const SOUTH_GEOMAGNETIC_POLE = { lon: 107.2, lat: -80.8 }
const NORTH_GEOMAGNETIC_POLE = { lon: -72.8, lat: 80.8 }

/**
 * Rotate given coordinates to center them around 90ºE 0º
 * This way the contours can be computed over an area that does not have any gap, e.g. without
 * rotating there would be a gap between 180ºE and 180ºW.
 */
const rotateCoordinates = (lon: number, lat: number): [number, number] => {
  // Convert to radians
  const lonRad = lon * Math.PI / 180
  const latRad = lat * Math.PI / 180

  // Pre-rotate by subtracting 90 degrees longitude
  const preLon = lonRad - Math.PI / 2

  // Cartesian coordinates
  const x = Math.cos(latRad) * Math.cos(preLon)
  const y = Math.cos(latRad) * Math.sin(preLon)
  const z = Math.sin(latRad)

  // Rotation
  const xPrime = z
  const yPrime = y
  const zPrime = -x

  // Convert back to spherical coordinates
  let newLon = Math.atan2(yPrime, xPrime) * 180 / Math.PI
  const newLat = Math.asin(zPrime) * 180 / Math.PI

  // Post-rotate by adding 90 degrees longitude
  newLon += 90

  // Adjust longitude to 0-360 range
  if (newLon < 0) newLon += 360

  return [ newLon, newLat ]
}

/**
 * Invert rotation from @see{rotateCoordinates}
 */
const invertRotation = (lon: number, lat: number): [number, number] => {
  // Convert to radians
  const lonRad = lon * Math.PI / 180
  const latRad = lat * Math.PI / 180

  // Pre-rotate by subtracting 90 degrees longitude
  const preLon = lonRad - Math.PI / 2

  // Cartesian coordinates
  const x = Math.cos(latRad) * Math.cos(preLon)
  const y = Math.cos(latRad) * Math.sin(preLon)
  const z = Math.sin(latRad)

  // Inverse rotation
  const xPrime = -z
  const yPrime = y
  const zPrime = x

  // Convert back to spherical coordinates
  let originalLon = Math.atan2(yPrime, xPrime) * 180 / Math.PI
  const originalLat = Math.asin(zPrime) * 180 / Math.PI

  // Post-rotate by adding 90 degrees longitude
  originalLon += 90

  // Adjust longitude to 0-360 range
  if (originalLon < 0) originalLon += 360

  return [ originalLon, originalLat ]
}

const rotateCoordinatesToGeomagneticSouthPole = (lon: number, lat: number): [number, number] => {
  // Convert inputs to radians
  const lonRad = lon * Math.PI / 180
  const latRad = lat * Math.PI / 180
  const centerLonRad = NORTH_GEOMAGNETIC_POLE.lon * Math.PI / 180
  const centerLatRad = SOUTH_GEOMAGNETIC_POLE.lat * Math.PI / 180
  const targetLonRad = SOUTH_GEOMAGNETIC_POLE.lon * Math.PI / 180
  const targetLatRad = SOUTH_GEOMAGNETIC_POLE.lat * Math.PI / 180

  // Convert to Cartesian coordinates
  const x = Math.cos(latRad) * Math.cos(lonRad)
  const y = Math.cos(latRad) * Math.sin(lonRad)
  const z = Math.sin(latRad)

  const centerX = Math.cos(centerLatRad) * Math.cos(centerLonRad)
  const centerY = Math.cos(centerLatRad) * Math.sin(centerLonRad)
  const centerZ = Math.sin(centerLatRad)

  const targetX = Math.cos(targetLatRad) * Math.cos(targetLonRad)
  const targetY = Math.cos(targetLatRad) * Math.sin(targetLonRad)
  const targetZ = Math.sin(targetLatRad)

  // Calculate the rotation axis as the cross product of center and target
  const axisX = centerY * targetZ - centerZ * targetY
  const axisY = centerZ * targetX - centerX * targetZ
  const axisZ = centerX * targetY - centerY * targetX

  // Normalize the rotation axis
  const axisLength = Math.sqrt(axisX ** 2 + axisY ** 2 + axisZ ** 2)
  const normAxisX = axisX / axisLength
  const normAxisY = axisY / axisLength
  const normAxisZ = axisZ / axisLength

  // Calculate the rotation angle (dot product of center and target)
  const dotProduct =
    centerX * targetX + centerY * targetY + centerZ * targetZ
  const angle = Math.acos(dotProduct) // Angle in radians

  // Perform the rotation using Rodrigues' rotation formula
  const cosAngle = Math.cos(angle)
  const sinAngle = Math.sin(angle)
  const oneMinusCos = 1 - cosAngle

  const xPrime =
    x * (cosAngle + normAxisX ** 2 * oneMinusCos) +
    y * (normAxisX * normAxisY * oneMinusCos - normAxisZ * sinAngle) +
    z * (normAxisX * normAxisZ * oneMinusCos + normAxisY * sinAngle)

  const yPrime =
    x * (normAxisY * normAxisX * oneMinusCos + normAxisZ * sinAngle) +
    y * (cosAngle + normAxisY ** 2 * oneMinusCos) +
    z * (normAxisY * normAxisZ * oneMinusCos - normAxisX * sinAngle)

  const zPrime =
    x * (normAxisZ * normAxisX * oneMinusCos - normAxisY * sinAngle) +
    y * (normAxisZ * normAxisY * oneMinusCos + normAxisX * sinAngle) +
    z * (cosAngle + normAxisZ ** 2 * oneMinusCos)

  // Convert back to spherical coordinates
  const newLon = Math.atan2(yPrime, xPrime) * 180 / Math.PI
  const newLat = Math.asin(zPrime) * 180 / Math.PI

  return [ newLon, newLat ]
}

/**
 * Given a number or array of thresholds (number of contours) and the matrix of points with score, generate d3 contours
 * Generate contours using d3.contourDensity
 * Remove 20º of Latitude in order to "stretch" the oval and favor contours that are thinner in the longitudinal axis
 * and do not connect over the poles.
 * Rotate the points coordinates so that the oval is computed over one contiguous zone to prevent gaps.
 * Add 90º of latitude in order to have all coordinates above 0
 * Then invert all said operations after contours have been computed
 *
 * @param thresholds
 * @param data [ lon, lat, score ]
 * @param ponderateWeight boolean to ponderate (square) the given weight (score in `data`) defaults to true. Helps pushing
 *                        high values that are "alone" to have a bigger contour.
 * @param bandwidth d3 contour bandwidth defaults to 2
 * @param cellSize d3 contour cell size defaults to 1
 */
export const getContours = (
  thresholds: Array<number> | number,
  data: Array<[number, number, number]>,
  ponderateWeight = true,
  bandwidth = 2,
  cellSize = 1
) => {
  const latShift = 20
  const contours = d3.contourDensity<[number, number, number]>()
    // Rotate coordinates to avoid 180º longitude gap. remove 40 or 20 latitude to "spread" the oval depending on souht/north
    // to favor thin oval
    .x(([ lon, lat ]) => rotateCoordinates(lon, lat < 0 ? lat + latShift : lat - latShift)[0])
    .y(([ lon, lat ]) => rotateCoordinates(lon, lat < 0 ? lat + latShift : lat - latShift)[1] + 90)
    .weight(d => (ponderateWeight ? Math.pow(d[2], 1.5) : d[2]))
    .size([ 360, 180 ])
    .cellSize(cellSize)
    .bandwidth(bandwidth)
    .thresholds(thresholds)([
      ...data,
      ...data.map(([ lon, lat, score ]) => [
        ...rotateCoordinatesToGeomagneticSouthPole(lon, -lat),
        score
      ])
    ] as Array<[number, number, number]>)

  return contours
    .map(d => ({
      ...d,
      coordinates: d.coordinates.map(c => c.map(p => p.map(([ lon, lat ]) => {
        const [ rLon, rLat ] = invertRotation(lon, lat - 90)
        return [
          rLon,
          rLat < 0 ? rLat - latShift : rLat + latShift
        ]
      })))
    }))
    .filter(d => d.value > 0) as Array<d3.ContourMultiPolygon>
}
