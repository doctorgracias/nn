import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { Line, OrbitControls, PerspectiveCamera, Html } from '@react-three/drei';
import * as THREE from 'three';

const Trajectory = ({ data, color = "cyan" }) => {
  const points = useMemo(() => {
    if (!Array.isArray(data) || data.length < 2) {
      return [new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0)];
    }

    const result = [];
    for (const p of data) {
      const x = parseFloat(p[1]);
      const y = parseFloat(p[2]);
      const z = parseFloat(p[3]);

      if (!isNaN(x) && !isNaN(y) && !isNaN(z) && isFinite(x) && isFinite(y) && isFinite(z)) {
        result.push(new THREE.Vector3(x, y, z));
      }
    }
    return result.length > 1 ? result : [new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0)];
  }, [data]);

  return <Line points={points} color={color} lineWidth={1.5} />;
};

export const Scene = ({ solution, prediction = [] }) => {
  return (
    <div style={{ width: '100%', height: '600px', background: '#000' }}>
      <Canvas>
        {/* Приблизил камеру (с 30 до 10), чтобы график был крупнее */}
        <PerspectiveCamera makeDefault position={[10, 10, 10]} />
        <OrbitControls />
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />
        
        <Trajectory data={solution} color="cyan" />
        {prediction.length > 0 && <Trajectory data={prediction} color="hotpink" />}
        
        {/* Основная сетка */}
        <gridHelper args={[20, 20, 0x888888, 0x444444]} />
        
        {/* Оси координат */}
        <axesHelper args={[5]} /> 

        {/* Подписи осей */}
        <Html position={[5.2, 0, 0]}><span style={{ color: 'red', fontWeight: 'bold' }}>X</span></Html>
        <Html position={[0, 5.2, 0]}><span style={{ color: 'green', fontWeight: 'bold' }}>Y</span></Html>
        <Html position={[0, 0, 5.2]}><span style={{ color: 'blue', fontWeight: 'bold' }}>Z</span></Html>
      </Canvas>
    </div>
  );
};