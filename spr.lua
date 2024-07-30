--!strict
--!native
---------------------------------------------------------------------
-- spr - Spring-driven motion library
--
-- Copyright (c) 2024 Fractality. All rights reserved.
-- Released under the MIT license.
--
-- Docs & license can be found at https://github.com/Fraktality/spr
--
-- API Summary:
--
-- spr.target(
--    Instance obj,
--    number dampingRatio,
--    number undampedFrequency,
--    dict<string, Variant> targetProperties)
--
--    Animates the given properties towardes the target values,
--    given damping ratio and undamped frequency.
--
--
-- spr.stop(
--    Instance obj[,
--    string property])
--
--    Stops the specified property on an Instance from animating.
--    If no property is specified, all properties of the Instance
--    will stop animating.
--
-- Visualizer: https://www.desmos.com/calculator/rzvw27ljh9
---------------------------------------------------------------------

local STRICT_RUNTIME_TYPES = true -- assert on parameter and property type mismatch
local SLEEP_OFFSET_SQ_LIMIT = (1/3840)^2 -- square of the offset sleep limit
local SLEEP_VELOCITY_SQ_LIMIT = 1e-2^2 -- square of the velocity sleep limit
local SLEEP_ROTATION_EPS = math.rad(0.01) -- rad
local SLEEP_ROTATION_VELOCITY = math.rad(0.1) -- rad/s
local EPS = 1e-5 -- epsilon for stability checks around pathological frequency/damping values
local AXIS_MATRIX_EPS = 1e-6 -- epsilon for converting from axis-angle to matrix

local RunService: RunService = game:GetService("RunService")

local pi = math.pi
local exp = math.exp
local sin = math.sin
local cos = math.cos
local min = math.min
local sqrt = math.sqrt
local round = math.round
local dot = Vector3.zero.Dot

local function magnitudeSq(vec: {number})
	local out = 0
	for _, v in vec do
		out += v^2
	end
	return out
end

local function distanceSq(vec0: {number}, vec1: {number})
	local out = 0
	for i0, v0 in vec0 do
		out += (vec1[i0] - v0)^2
	end
	return out
end

local function areRotationsClose(c0: CFrame, c1: CFrame)
	local rx = dot(c0.XVector, c1.XVector)
	local ry = dot(c0.YVector, c1.YVector)
	local rz = dot(c0.ZVector, c1.ZVector)
	local trace = rx + ry + rz
	return trace > 1 + 2*cos(SLEEP_ROTATION_EPS)
end

type TypeMetadata<T> = {
	springType: (dampingRatio: number, frequency: number, pos: number, typedat: TypeMetadata<T>, rawTarget: T) -> LinearSpring<T>,
	toIntermediate: (T) -> {number},
	fromIntermediate: ({number}) -> T,
}

-- Spring for an array of linear values
local LinearSpring = {}

type LinearSpring<T> = typeof(setmetatable({} :: {
	d: number,
	f: number,
	g: {number},
	p: {number},
	v: {number},
	typedat: TypeMetadata<T>,
	rawTarget: T,
}, LinearSpring))

do
	LinearSpring.__index = LinearSpring

	function LinearSpring.new<T>(dampingRatio: number, frequency: number, pos: T, rawGoal: T, typedat)
		local linearPos = typedat.toIntermediate(pos)
		return setmetatable(
			{
				d = dampingRatio,
				f = frequency,
				g = linearPos,
				p = linearPos,
				v = table.create(#linearPos, 0),
				typedat = typedat,
				rawGoal = rawGoal
			},
			LinearSpring
		)
	end

	function LinearSpring.setGoal<T>(self, goal: T)
		self.rawGoal = goal
		self.g = self.typedat.toIntermediate(goal)
	end

	function LinearSpring.setDampingRatio<T>(self: LinearSpring<T>, dampingRatio: number)
		self.d = dampingRatio
	end

	function LinearSpring.setFrequency<T>(self: LinearSpring<T>, frequency: number)
		self.f = frequency
	end

	function LinearSpring.canSleep<T>(self)
		if magnitudeSq(self.v) > SLEEP_VELOCITY_SQ_LIMIT then
			return false
		end

		if distanceSq(self.p, self.g) > SLEEP_OFFSET_SQ_LIMIT then
			return false
		end

		return true
	end

	function LinearSpring.step<T>(self: LinearSpring<T>, dt: number)
		-- Advance the spring simulation by dt seconds.
		-- Take the damped harmonic oscillator ODE:
		--    f^2*(X[t] - g) + 2*d*f*X'[t] + X''[t] = 0
		-- Where X[t] is position at time t, g is target position,
		-- f is undamped angular frequency, and d is damping ratio.
		-- Apply constant initial conditions:
		--    X[0] = p0
		--    X'[0] = v0
		-- Solve the IVP to get analytic expressions for X[t] and X'[t].
		-- The solution takes one of three forms for 0<=d<1, d=1, and d>1

		local d = self.d
		local f = self.f*2*pi -- Hz -> Rad/s
		local g = self.g
		local p = self.p
		local v = self.v

		if d == 1 then -- critically damped
			local q = exp(-f*dt)
			local w = dt*q

			local c0 = q + w*f
			local c2 = q - w*f
			local c3 = w*f*f

			for idx = 1, #p do
				local o = p[idx] - g[idx]
				p[idx] = o*c0 + v[idx]*w + g[idx]
				v[idx] = v[idx]*c2 - o*c3
			end

		elseif d < 1 then -- underdamped
			local q = exp(-d*f*dt)
			local c = sqrt(1 - d*d)

			local i = cos(dt*f*c)
			local j = sin(dt*f*c)

			-- Damping ratios approaching 1 can cause division by very small numbers.
			-- To mitigate that, group terms around z=j/c and find an approximation for z.
			-- Start with the definition of z:
			--    z = sin(dt*f*c)/c
			-- Substitute a=dt*f:
			--    z = sin(a*c)/c
			-- Take the Maclaurin expansion of z with respect to c:
			--    z = a - (a^3*c^2)/6 + (a^5*c^4)/120 + O(c^6)
			--    z ≈ a - (a^3*c^2)/6 + (a^5*c^4)/120
			-- Rewrite in Horner form:
			--    z ≈ a + ((a*a)*(c*c)*(c*c)/20 - c*c)*(a*a*a)/6

			local z
			if c > EPS then
				z = j/c
			else
				local a = dt*f
				z = a + ((a*a)*(c*c)*(c*c)/20 - c*c)*(a*a*a)/6
			end

			-- Frequencies approaching 0 present a similar problem.
			-- We want an approximation for y as f approaches 0, where:
			--    y = sin(dt*f*c)/(f*c)
			-- Substitute b=dt*c:
			--    y = sin(b*c)/b
			-- Now reapply the process from z.

			local y
			if f*c > EPS then
				y = j/(f*c)
			else
				local b = f*c
				y = dt + ((dt*dt)*(b*b)*(b*b)/20 - b*b)*(dt*dt*dt)/6
			end

			for idx = 1, #p do
				local o = p[idx] - g[idx]
				p[idx] = (o*(i + z*d) + v[idx]*y)*q + g[idx]
				v[idx] = (v[idx]*(i - z*d) - o*(z*f))*q
			end

		else -- overdamped
			local c = sqrt(d*d - 1)

			local r1 = -f*(d + c)
			local r2 = -f*(d - c)

			local ec1 = exp(r1*dt)
			local ec2 = exp(r2*dt)

			for idx = 1, #p do
				local o = p[idx] - g[idx]
				local co2 = (v[idx] - o*r1)/(2*f*c)
				local co1 = ec1*(o - co2)

				p[idx] = co1 + co2*ec2 + g[idx]
				v[idx] = co1*r1 + co2*ec2*r2
			end
		end

		return self.typedat.fromIntermediate(self.p)
	end
end

local RotationSpring = {} 

type RotationSpring = typeof(setmetatable({} :: {
	d: number,
	f: number,
	g: CFrame,
	p: CFrame,
	v: Vector3,
}, RotationSpring))

do
	RotationSpring.__index = RotationSpring

	local function matrixToAxis(m: CFrame)
		local axis, angle = m:ToAxisAngle()
		return axis*angle
	end

	local function axisToMatrix(v: Vector3)
		local mag = v.Magnitude
		if mag > AXIS_MATRIX_EPS then
			return CFrame.fromAxisAngle(v.Unit, mag)
		end
		return CFrame.identity
	end

	function RotationSpring.new(d: number, f: number, p: CFrame, g: CFrame)
		return setmetatable(
			{
				d = d,
				f = f,
				g = g,
				p = p,
				v = Vector3.zero
			},
			RotationSpring
		)
	end

	function RotationSpring.setGoal(self: RotationSpring, value: CFrame)
		self.g = value
	end

	function RotationSpring.setDampingRatio(self: RotationSpring, dampingRatio: number)
		self.d = dampingRatio
	end

	function RotationSpring.setFrequency(self: RotationSpring, frequency: number)
		self.f = frequency
	end

	function RotationSpring.canSleep(self: RotationSpring)
		local sleepP = areRotationsClose(self.p, self.g)
		local sleepV = self.v.Magnitude < SLEEP_ROTATION_VELOCITY
		return sleepP and sleepV
	end

	function RotationSpring.step(self: RotationSpring, dt: number): CFrame
		local d = self.d
		local f = self.f*2*pi
		local g = self.g
		local p0 = self.p
		local v0 = self.v

		local offset = matrixToAxis(p0*g:Inverse())
		local decay = exp(-d*f*dt)

		local pt: CFrame
		local vt: Vector3

		if d == 1 then -- critically damped
			pt = axisToMatrix((offset*(1 + f*dt) + v0*dt)*decay)*g
			vt = (v0*(1 - dt*f) - offset*(dt*f*f))*decay

		elseif d < 1 then -- underdamped
			local c = sqrt(1 - d*d)

			local i = cos(dt*f*c)
			local j = sin(dt*f*c)

			local y = j/(f*c)
			local z = j/c

			pt = axisToMatrix((offset*(i + z*d) + v0*y)*decay)*g
			vt = (v0*(i - z*d) - offset*(z*f))*decay

		else -- overdamped
			local c = sqrt(d*d - 1)

			local r1 = -f*(d + c)
			local r2 = -f*(d - c)

			local co2 = (v0 - offset*r1)/(2*f*c)
			local co1 = offset - co2

			local e1 = co1*exp(r1*dt)
			local e2 = co2*exp(r2*dt)

			pt = axisToMatrix(e1 + e2)*g
			vt = e1*r1 + e2*r2
		end

		self.p = pt
		self.v = vt

		return pt
	end
end

-- Defined early to be used by CFrameSpring
local typeMetadata_Vector3 = {
	springType = LinearSpring.new,

	toIntermediate = function(value)
		return {value.X, value.Y, value.Z}
	end,

	fromIntermediate = function(value: {number})
		return Vector3.new(value[1], value[2], value[3])
	end,
}

-- Encapsulates a CFrame - Separates translation from rotation
local CFrameSpring = {}
do
	CFrameSpring.__index = CFrameSpring

	function CFrameSpring.new(
		dampingRatio: number,
		frequency: number,
		valueCurrent: CFrame,
		valueGoal: CFrame,
		_: any
	)
		return setmetatable(
			{
				rawGoal = valueGoal,
				_position = LinearSpring.new(dampingRatio, frequency, valueCurrent.Position, valueGoal.Position, typeMetadata_Vector3),
				_rotation = RotationSpring.new(dampingRatio, frequency, valueCurrent.Rotation, valueGoal.Rotation)
			},
			CFrameSpring
		)
	end

	function CFrameSpring:setGoal(value: CFrame)
		self.rawGoal = value
		self._position:setGoal(value.Position)
		self._rotation:setGoal(value.Rotation)
	end

	function CFrameSpring:setDampingRatio(value: number)
		self._position:setDampingRatio(value)
		self._rotation:setDampingRatio(value)
	end

	function CFrameSpring:setFrequency(value: number)
		self._position:setFrequency(value)
		self._rotation:setFrequency(value)
	end

	function CFrameSpring:canSleep()
		return self._position:canSleep() and self._rotation:canSleep()
	end

	function CFrameSpring:step(dt): CFrame
		local p: Vector3 = self._position:step(dt)
		local r: CFrame = self._rotation:step(dt)
		return r + p
	end
end

-- Color conversions
local rgbToOklab
local oklabToRgb
do
	local function cbrt(x)
		return math.sign(x)*math.abs(x)^(1/3)
	end

	function rgbToOklab(value: Color3): {number}
		local r, g, b = value.R, value.G, value.B

		local l = cbrt(0.4122214708*r + 0.5363325363*g + 0.0514459929*b)
		local m = cbrt(0.2119034982*r + 0.6806995451*g + 0.1073969566*b)
		local s = cbrt(0.0883024619*r + 0.2817188376*g + 0.6299787005*b)

		return {
			0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
			1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
			0.0259040371*l + 0.7827717662*m - 0.8086757660*s}
	end

	function oklabToRgb(value: {number}): Color3
		local L, a, b = value[1], value[2], value[3]

		local l = (L + 0.3963377774*a + 0.2158037573*b)^3
		local m = (L - 0.1055613458*a - 0.0638541728*b)^3
		local s = (L - 0.0894841775*a - 1.2914855480*b)^3
		
		if l < 0 or m < 0 or s < 0 then
			print(math.sign(l), math.sign(m), math.sign(s))
		end

		local r = 4.0767416621*l - 3.3077115913*m + 0.2309699292*s
		local g = -1.2684380046*l + 2.6097574011*m - 0.3413193965*s
		local b = -0.0041960863*l - 0.7034186147*m + 1.7076147010*s
		
		r = math.clamp(r, 0, 1)
		g = math.clamp(g, 0, 1)
		b = math.clamp(b, 0, 1)

		return Color3.new(r, g, b)
	end
end

-- Type definitions
-- Transforms Roblox types into intermediate types, converting
-- between spaces as necessary to preserve perceptual linearity
local typeMetadata = {
	boolean = {
		springType = LinearSpring.new,

		toIntermediate = function(value)
			return {value and 1 or 0}
		end,

		fromIntermediate = function(value)
			return value[1] >= 0.5
		end,
	},

	number = {
		springType = LinearSpring.new,

		toIntermediate = function(value)
			return {value}
		end,

		fromIntermediate = function(value)
			return value[1]
		end,
	},

	NumberRange = {
		springType = LinearSpring.new,

		toIntermediate = function(value)
			return {value.Min, value.Max}
		end,

		fromIntermediate = function(value)
			return NumberRange.new(value[1], value[2])
		end,
	},

	UDim = {
		springType = LinearSpring.new,

		toIntermediate = function(value)
			return {value.Scale, value.Offset}
		end,

		fromIntermediate = function(value: {number})
			return UDim.new(value[1], round(value[2]))
		end,
	},

	UDim2 = {
		springType = LinearSpring.new,

		toIntermediate = function(value)
			local x = value.X
			local y = value.Y
			return {x.Scale, x.Offset, y.Scale, y.Offset}
		end,

		fromIntermediate = function(value: {number})
			return UDim2.new(value[1], round(value[2]), value[3], round(value[4]))
		end,
	},

	Vector2 = {
		springType = LinearSpring.new,

		toIntermediate = function(value)
			return {value.X, value.Y}
		end,

		fromIntermediate = function(value: {number})
			return Vector2.new(value[1], value[2])
		end,
	},

	Vector3 = typeMetadata_Vector3,

	Color3 = {
		springType = LinearSpring.new,
		toIntermediate = rgbToOklab,
		fromIntermediate = oklabToRgb,
	},

	-- Only interpolates start and end keypoints
	ColorSequence = {
		springType = LinearSpring.new,

		toIntermediate = function(value)
			local keypoints = value.Keypoints
			local from = rgbToOklab(keypoints[1].Value)
			local to = rgbToOklab(keypoints[#keypoints].Value)
			return {
				from[1], from[2], from[3],
				to[1], to[2], to[3]}
		end,

		fromIntermediate = function(value: {})
			return ColorSequence.new(
				oklabToRgb{value[1], value[2], value[3]},
				oklabToRgb{value[4], value[5], value[6]})
		end,
	},

	CFrame = {
		springType = CFrameSpring.new,
		toIntermediate = error, -- custom (CFrameSpring)
		fromIntermediate = error, -- custom (CFrameSpring)
	}
}

type PropertyOverride = {
	[string]: {
		class: string,
		get: (any)->(),
		set: (any, any)->(),
	}
}

local PSEUDO_PROPERTIES: PropertyOverride = {
	Pivot = {
		class = "PVInstance",
		get = function(inst: PVInstance)
			return inst:GetPivot()
		end,
		set = function(inst: PVInstance, value: CFrame)
			inst:PivotTo(value)
		end
	},
	Scale = {
		class = "Model",
		get = function(inst: Model)
			return inst:GetScale()
		end,
		set = function(inst: Model, value: number)
			inst:ScaleTo(value)
		end
	}
}

local function getProperty(instance: Instance, property: string): any
	local override = PSEUDO_PROPERTIES[property]
	if override and instance:IsA(override.class) then
		return override.get(instance)
	else
		return (instance :: any)[property]
	end
end

local function setProperty(instance: Instance, property: string, value: unknown)
	local override = PSEUDO_PROPERTIES[property]
	if override and instance:IsA(override.class) then
		override.set(instance, value)
	else
		(instance :: any)[property] = value
	end
end

-- Frame loop
local springStates_other: {[Instance]: {[string]: any}} = {} -- {[instance] = {[property] = spring}
local springStates_render: {[Instance]: {[string]: any}} = {} -- {[instance] = {[property] = spring}
local completedCallbacks: {[Instance]: {()->()}} = {}

local function processSprings(springStates: typeof(springStates_other), dt: number)
	for instance, state in springStates do
		for propName, spring in state do
			if spring:canSleep() then
				state[propName] = nil
				setProperty(instance, propName, spring.rawGoal)
			else
				setProperty(instance, propName, spring:step(dt))
			end
		end

		if not next(state) then
			springStates[instance] = nil

			-- trigger completed callbacks when all properties finish animating
			local callbackList = completedCallbacks[instance]
			if callbackList then
				-- flush callback list before we run any callbacks in case
				-- one of the callbacks recursively adds another callback
				completedCallbacks[instance] = nil

				for _, callback in callbackList do
					task.spawn(callback)
				end
			end
		end
	end
end

RunService.PreSimulation:Connect(function(dt)
	processSprings(springStates_other, dt)
end)

RunService.PostSimulation:Connect(function(dt)
	processSprings(springStates_render, dt)
end)

local function assertType(argNum: number, fnName: string, expectedType: string, value: unknown)
	if not expectedType:find(typeof(value)) then
		error(`bad argument #{argNum} to {fnName} ({expectedType} expected, got {typeof(value)})`, 3)
	end
end

-- API
local spr = {}

function spr.target(instance: Instance, dampingRatio: number, frequency: number, properties: {[string]: any})
	if STRICT_RUNTIME_TYPES then
		assertType(1, "spr.target", "Instance", instance)
		assertType(2, "spr.target", "number", dampingRatio)
		assertType(3, "spr.target", "number", frequency)
		assertType(4, "spr.target", "table", properties)
	end

	if dampingRatio ~= dampingRatio or dampingRatio < 0 then
		error(("expected damping ratio >= 0; got %.2f"):format(dampingRatio), 2)
	end

	if frequency ~= frequency or frequency < 0 then
		error(("expected undamped frequency >= 0; got %.2f"):format(frequency), 2)
	end

	local targetRecord = (if instance:IsA("Camera") then springStates_render else springStates_other) :: {[Instance]: {[string]: any}}

	local state = targetRecord[instance]
	if not state then
		state = {}
		targetRecord[instance] = state
	end

	for propName, propTarget in properties do
		local propValue = getProperty(instance, propName)

		if STRICT_RUNTIME_TYPES and typeof(propTarget) ~= typeof(propValue) then
			error(`bad property {propName} to spr.target ({typeof(propValue)} expected, got {typeof(propTarget)})`, 2)
		end

		-- Special case infinite frequency for an instantaneous change
		if frequency == math.huge then
			setProperty(instance, propName, propTarget)
			state[propName] = nil
			continue
		end

		local spring = state[propName]
		if not spring then
			local md = typeMetadata[typeof(propTarget)]
			if not md then
				error("unsupported type: " .. typeof(propTarget), 2)
			end

			spring = md.springType(dampingRatio, frequency, propValue, propTarget, md)
			state[propName] = spring
		end

		spring:setGoal(propTarget)
		spring:setDampingRatio(dampingRatio)
		spring:setFrequency(frequency)
	end

	if not next(state) then
		targetRecord[instance] = nil
	end
end

function spr.stop(instance: Instance, property: string?)
	if STRICT_RUNTIME_TYPES then
		assertType(1, "spr.stop", "Instance", instance)
		assertType(2, "spr.stop", "string|nil", property)
	end

	if property then
		local state = springStates_other[instance] or springStates_render[instance]
		if state then
			state[property] = nil
		end
	else
		springStates_other[instance] = nil
		springStates_render[instance] = nil
	end
end

function spr.completed(instance: Instance, callback: ()->())
	if STRICT_RUNTIME_TYPES then
		assertType(1, "spr.completed", "Instance", instance)
		assertType(2, "spr.completed", "function", callback)
	end

	local callbackList = completedCallbacks[instance]
	if callbackList then
		table.insert(callbackList, callback)
	else
		completedCallbacks[instance] = {callback}
	end
end

return table.freeze(spr)
