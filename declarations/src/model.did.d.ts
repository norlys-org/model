import type { Principal } from '@dfinity/principal';
import type { ActorMethod } from '@dfinity/agent';
import type { IDL } from '@dfinity/candid';

export interface ObservationVector {
  'i' : number,
  'j' : number,
  'k' : number,
  'lat' : number,
  'lon' : number,
}
export interface _SERVICE {
  'a_add_authorized_user' : ActorMethod<[Principal], undefined>,
  'a_initialize_authorized_user' : ActorMethod<[Principal], undefined>,
  'a_list_authorized_users' : ActorMethod<[], Array<Principal>>,
  'a_remove_authorized_user' : ActorMethod<[Principal], undefined>,
  'm_fit_obs' : ActorMethod<[Array<ObservationVector>], undefined>,
  'm_fit_pred' : ActorMethod<[], undefined>,
  'm_predict' : ActorMethod<[boolean], undefined>,
  'm_scores' : ActorMethod<[], Uint16Array | number[]>,
}
export declare const idlFactory: IDL.InterfaceFactory;
export declare const init: (args: { IDL: typeof IDL }) => IDL.Type[];
